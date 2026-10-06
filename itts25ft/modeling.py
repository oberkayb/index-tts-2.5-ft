"""
Model construction, surgical parameter selection and checkpoint export.

Adding a language to IndexTTS-2.5 is mostly an exercise in *not* breaking the
five languages it already speaks (zh, en, ja, es, ar) or the cross-lingual and
timbre/emotion disentanglement that the 2.5 release is built around.  Three
mechanisms in this module do that work:

  * :func:`init_language_row` seeds the new ``lang_embedding`` row from a
    trained language instead of leaving it at its random initialisation, which
    is what the shipped checkpoint holds for every untrained row.
  * :class:`LanguageEmbeddingGradMask` zeroes the gradient of every *other*
    language row, so the languages you are not training are provably unchanged.
    :class:`EmbeddingRowGradMask` applies the same idea to the text embedding
    (see :func:`select_text_rows`).
  * :func:`inject_lora` keeps the GPT body frozen and learns low-rank deltas,
    the cheapest reliable defence against catastrophic forgetting.  Adapters
    merge back into the base weights on export, so the result is a plain
    ``gpt.pth`` that stock ``infer_v2_5.py`` loads with no code changes.
"""

from __future__ import annotations

import fnmatch
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

def build_gpt(
    cfg,
    model_dir: str | Path,
    device: str | torch.device = "cuda",
    checkpoint: Optional[str | Path] = None,
    strict_check: bool = True,
):
    """Instantiate ``UnifiedVoice`` in 2.5 (campplus) mode and load weights."""
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint

    model = UnifiedVoice(**cfg.gpt, spk_cond_mode="campplus")

    ckpt_path = Path(checkpoint) if checkpoint else Path(model_dir) / cfg.gpt_checkpoint
    if not ckpt_path.is_file():
        raise FileNotFoundError("GPT checkpoint not found: " + str(ckpt_path))
    load_checkpoint(model, str(ckpt_path))

    if strict_check:
        if not hasattr(model, "spk_emb_proj"):
            raise RuntimeError(
                "Model has no spk_emb_proj: this is not an IndexTTS-2.5 style model. "
                "Check that --model-dir points at IndexTTS-2.5 weights."
            )
        if not hasattr(model, "lang_embedding"):
            raise RuntimeError("Model has no lang_embedding: not an IndexTTS-2.5 checkpoint.")

    return model.to(device)


def validate_text_vocab(model, max_token_id: int) -> None:
    """Fail loudly if the tiktoken ids overflow the text embedding table."""
    capacity = model.text_embedding.num_embeddings
    if max_token_id >= capacity:
        raise ValueError(
            "Text token id " + str(max_token_id) + " exceeds the model's text vocabulary ("
            + str(capacity) + "). The 2.5 checkpoint and the tokenizer are out of sync; "
            "verify that checkpoints/ holds the 2.5 tiktoken vocabulary."
        )


# --------------------------------------------------------------------------- #
# Language embedding
# --------------------------------------------------------------------------- #

def init_language_row(
    model,
    target_lang_id: int,
    source_lang_id: Optional[int] = None,
    noise: float = 0.0,
) -> str:
    """Seed the target ``lang_embedding`` row.

    The shipped checkpoint only trained rows for zh/en/ja/es/ar; every other row
    is untrained noise that weight decay shrank to a norm of ~0.17 (trained rows
    sit at 1-2).  Starting from a
    trained row (a typologically close one, e.g. Spanish for Turkish) converges
    noticeably faster than starting from that noise.
    """
    emb = model.lang_embedding
    if target_lang_id >= emb.num_embeddings:
        raise ValueError(
            "lang_id " + str(target_lang_id) + " is out of range for lang_embedding with "
            + str(emb.num_embeddings) + " rows."
        )
    if source_lang_id is None:
        return "kept the checkpoint's existing row (random init)"

    with torch.no_grad():
        src = emb.weight[source_lang_id].detach().clone()
        if noise > 0:
            src = src + torch.randn_like(src) * noise * src.std()
        emb.weight[target_lang_id].copy_(src)
    return "copied row " + str(source_lang_id) + " -> row " + str(target_lang_id)


class EmbeddingRowGradMask:
    """Restrict an embedding table's updates to a whitelist of rows.

    Registered as a tensor hook, so it applies to the gradient itself and works
    identically under AMP, gradient accumulation and any optimizer.  Masked rows
    keep exactly zero gradient, so AdamW never moves them as long as their
    parameter group has ``weight_decay=0`` (see :func:`parameter_groups`).
    """

    def __init__(self, weight: torch.Tensor, trainable_rows: Sequence[int], name: str = "embedding") -> None:
        self.rows = sorted(set(int(r) for r in trainable_rows))
        if not weight.requires_grad:
            raise RuntimeError(
                name + ".weight is frozen; register the mask *after* "
                "apply_trainable_spec() so the hook has a gradient to mask."
            )
        mask = torch.zeros(weight.shape[0], 1, dtype=weight.dtype)
        if self.rows:
            mask[self.rows] = 1.0
        self._mask = mask
        self._handle = weight.register_hook(self._apply)

    def _apply(self, grad: torch.Tensor) -> torch.Tensor:
        if self._mask.device != grad.device or self._mask.dtype != grad.dtype:
            self._mask = self._mask.to(device=grad.device, dtype=grad.dtype)
        return grad * self._mask

    def remove(self) -> None:
        self._handle.remove()


class LanguageEmbeddingGradMask(EmbeddingRowGradMask):
    """Restrict ``lang_embedding`` updates to a whitelist of rows."""

    def __init__(self, model, trainable_rows: Sequence[int]) -> None:
        super().__init__(model.lang_embedding.weight, trainable_rows, name="lang_embedding")


# --------------------------------------------------------------------------- #
# Text embedding rows
# --------------------------------------------------------------------------- #

#: Row norm below which a text-embedding row counts as untrained.  Measured on
#: the shipped 2.5 checkpoint: never-used tokens (``<|haw|>``, timestamp and
#: ``SPECIAL_TOKEN_*`` rows) sit at ~0.10, whole-word Turkish tokens such as
#: `` kaç`` / `` evet`` / `` bugün`` at 0.10-0.39, while tokens the base model
#: actually learned (English, Chinese, Spanish, shared sub-word pieces) sit at
#: 1.0-2.9.
UNTRAINED_ROW_NORM = 0.5


def select_text_rows(
    model,
    token_counts: Dict[int, int],
    min_count: int = 1,
    max_base_norm: float = UNTRAINED_ROW_NORM,
) -> Tuple[List[int], Dict[str, object]]:
    """Pick the ``text_embedding`` rows a new language needs trained.

    tiktoken folds frequent words *with* their leading space into one token
    (`` kaç``, `` evet``).  If the base model never saw the language, those rows
    are near-zero, so the GPT cannot tell such words apart and swaps one short
    word for another.  We train exactly the rows the corpus uses that the base
    model left untrained; rows other languages rely on stay frozen.

    ``max_base_norm <= 0`` disables the norm filter (train every used row).
    Call this on the *base* weights, before any row is re-initialised.
    """
    norms = model.text_embedding.weight.detach().float().norm(dim=1).cpu()
    capacity = norms.shape[0]

    rows: List[int] = []
    skipped_trained = skipped_rare = 0
    covered = total = 0
    for token_id, count in token_counts.items():
        total += count
        if token_id >= capacity:
            continue
        if count < min_count:
            skipped_rare += 1
            continue
        if max_base_norm > 0 and norms[token_id].item() >= max_base_norm:
            skipped_trained += 1
            continue
        rows.append(int(token_id))
        covered += count

    rows.sort()
    stats: Dict[str, object] = {
        "used_ids": len(token_counts),
        "selected": len(rows),
        "skipped_already_trained": skipped_trained,
        "skipped_below_min_count": skipped_rare,
        "occurrences_on_selected": covered,
        "occurrences_total": total,
    }
    return rows, stats


#: Letters the base model never learned, mapped to Latin spellings of a close
#: sound.  Only used to find *trained* pieces to seed an untrained row from.
_SEED_TRANSLIT = str.maketrans({
    "ç": "ch", "ş": "sh", "ğ": "", "ı": "i", "ö": "o", "ü": "u",
    "â": "a", "î": "i", "û": "u",
})


def init_text_rows_from_subwords(
    model,
    rows: Sequence[int],
    tokenizer,
    max_base_norm: float = UNTRAINED_ROW_NORM,
) -> Dict[str, int]:
    """Seed each untrained row with the mean of trained sub-word pieces.

    `` kaç`` is split into pieces the base model already reads (quoting a word
    forces such a split and fixes the pronunciation), so the merged token starts
    near a plausible sound instead of next to every other untrained word.  When
    a piece is itself untrained (``ç``, ``ı``, ``ş``), the word is respelled with
    :data:`_SEED_TRANSLIT` (`` kaç`` -> ``kach``) to find trained pieces.  This
    is only a starting point; training moves the rows.  Control tokens and rows
    with no trained pieces keep their current value.
    """
    weight = model.text_embedding.weight
    norms = weight.detach().float().norm(dim=1)
    encoding = getattr(tokenizer, "encoding", tokenizer)
    threshold = max(max_base_norm, 0.0)

    def trained(token_id: int) -> bool:
        return token_id < norms.shape[0] and norms[token_id].item() >= threshold

    stats = {"initialised": 0, "kept_special": 0, "kept_no_pieces": 0}
    with torch.no_grad():
        for token_id in rows:
            try:
                raw = encoding.decode_single_token_bytes(token_id)
                text = raw.decode("utf-8")
            except Exception:  # noqa: BLE001 - partial UTF-8 byte tokens
                stats["kept_no_pieces"] += 1
                continue
            if text.startswith("<|") and text.endswith("|>"):
                stats["kept_special"] += 1
                continue

            core = text.strip()
            pieces: List[int] = []
            for spelling in (core, core.translate(_SEED_TRANSLIT)):
                if not spelling:
                    continue
                candidate = encoding.encode(spelling, allowed_special=set())
                if candidate == [token_id] or len(candidate) <= 1:
                    candidate = [p for ch in spelling for p in encoding.encode(ch, allowed_special=set())]
                candidate = [p for p in candidate if p != token_id]
                if candidate and all(trained(p) for p in candidate):
                    pieces = candidate
                    break
                pieces = pieces or [p for p in candidate if trained(p)]
            if not pieces:
                stats["kept_no_pieces"] += 1
                continue

            seed = weight[torch.tensor(pieces, device=weight.device)].float().mean(dim=0)
            weight[token_id].copy_(seed.to(weight.dtype))
            stats["initialised"] += 1
    return stats


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #

class LoRAConv1D(nn.Module):
    """Low-rank adapter around HuggingFace's ``Conv1D`` (weight is [in, out])."""

    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        in_features, out_features = base.weight.shape
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # delta starts at exactly zero
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = (self.dropout(x) @ self.lora_A.to(x.dtype)) @ self.lora_B.to(x.dtype)
        return out + delta * self.scaling

    @torch.no_grad()
    def merge(self) -> nn.Module:
        """Fold the adapter into the base weight and return the plain module."""
        delta = (self.lora_A @ self.lora_B) * self.scaling
        self.base.weight.add_(delta.to(self.base.weight.dtype))
        return self.base


def inject_lora(
    model,
    rank: int = 32,
    alpha: float = 64.0,
    dropout: float = 0.0,
    targets: Sequence[str] = ("c_attn", "c_proj", "c_fc"),
    last_n_layers: int = 0,
) -> int:
    """Wrap the GPT block projections with LoRA. Returns the module count.

    ``last_n_layers=0`` adapts every block; a positive value restricts adaptation
    to the top N blocks, which is cheaper and even gentler on the base model.
    """
    blocks = list(model.gpt.h)
    if last_n_layers > 0:
        blocks = blocks[-last_n_layers:]

    count = 0
    for block in blocks:
        for parent in (block.attn, block.mlp):
            for name in targets:
                child = getattr(parent, name, None)
                if child is None or isinstance(child, LoRAConv1D):
                    continue
                if not hasattr(child, "weight") or child.weight.ndim != 2:
                    continue
                setattr(parent, name, LoRAConv1D(child, rank, alpha, dropout))
                count += 1
    return count


@torch.no_grad()
def merge_lora(model) -> int:
    """Fold every adapter back into its base module, in place."""
    merged = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, LoRAConv1D):
                setattr(module, name, child.merge())
                merged += 1
    return merged


def has_lora(model) -> bool:
    return any(isinstance(m, LoRAConv1D) for m in model.modules())


# --------------------------------------------------------------------------- #
# Freezing / parameter groups
# --------------------------------------------------------------------------- #

#: Modules that carry the 2.5 behaviours we want to preserve verbatim.
#: The emotion branch defines timbre/emotion disentanglement; spk_emb_proj is
#: the speaker interface that makes cross-lingual cloning work at all.
PRESERVE_PATTERNS: Tuple[str, ...] = (
    "emo_conditioning_encoder.*",
    "emo_perceiver_encoder.*",
    "emovec_layer.*",
    "emo_layer.*",
    "spk_emb_proj.*",
)


def set_trainable(model, train_patterns: Iterable[str], freeze_patterns: Iterable[str] = ()) -> Dict[str, int]:
    """Mark parameters trainable by glob pattern; freeze wins over train."""
    train_patterns = list(train_patterns)
    freeze_patterns = list(freeze_patterns)

    for param in model.parameters():
        param.requires_grad = False

    stats = {"trainable": 0, "frozen": 0}
    for name, param in model.named_parameters():
        wanted = any(fnmatch.fnmatch(name, pat) for pat in train_patterns)
        blocked = any(fnmatch.fnmatch(name, pat) for pat in freeze_patterns)
        param.requires_grad = bool(wanted and not blocked)
        stats["trainable" if param.requires_grad else "frozen"] += param.numel()
    return stats


@dataclass
class TrainableSpec:
    """Declarative description of what gets updated during finetuning."""

    mode: str = "lora"                       # lora | partial | full
    lora_rank: int = 32
    lora_alpha: float = 64.0
    lora_dropout: float = 0.0
    lora_last_n_layers: int = 0
    train_last_n_layers: int = 6             # for mode=partial
    train_text_embedding: bool = False       # whole table + positions (everything moves)
    train_text_rows: bool = False            # table only; rows restricted by EmbeddingRowGradMask
    train_heads: bool = True
    preserve_emotion: bool = True
    extra_train: List[str] = field(default_factory=list)
    extra_freeze: List[str] = field(default_factory=list)


def apply_trainable_spec(model, spec: TrainableSpec) -> Dict[str, object]:
    """Configure the model per ``spec`` and report what ended up trainable."""
    info: Dict[str, object] = {"mode": spec.mode, "lora_modules": 0}
    if spec.train_text_embedding and spec.train_text_rows:
        raise ValueError(
            "train_text_embedding (whole table) and train_text_rows (masked rows) "
            "are mutually exclusive; pick one."
        )

    # The language embedding is always trained - it is the whole point.
    patterns: List[str] = ["lang_embedding.weight"]

    if spec.mode == "lora":
        info["lora_modules"] = inject_lora(
            model,
            rank=spec.lora_rank,
            alpha=spec.lora_alpha,
            dropout=spec.lora_dropout,
            last_n_layers=spec.lora_last_n_layers,
        )
        patterns += ["*.lora_A", "*.lora_B"]
    elif spec.mode == "partial":
        n_layers = len(model.gpt.h)
        first = max(0, n_layers - spec.train_last_n_layers)
        patterns += ["gpt.h." + str(i) + ".*" for i in range(first, n_layers)]
        patterns += ["final_norm.*"]
    elif spec.mode == "full":
        patterns += ["*"]
    else:
        raise ValueError("Unknown trainable mode: " + repr(spec.mode))

    if spec.train_heads and spec.mode != "full":
        patterns += ["mel_head.*", "text_head.*"]
    if spec.train_text_embedding and spec.mode != "full":
        patterns += ["text_embedding.weight", "text_pos_embedding.*"]
    if spec.train_text_rows and spec.mode != "full":
        patterns += ["text_embedding.weight"]

    patterns += list(spec.extra_train)

    freeze = list(spec.extra_freeze)
    if spec.preserve_emotion:
        freeze += list(PRESERVE_PATTERNS)

    stats = set_trainable(model, patterns, freeze)
    info.update(stats)
    info["patterns"] = patterns
    info["freeze_patterns"] = freeze
    return info


def parameter_groups(
    model,
    base_lr: float,
    lang_lr_multiplier: float = 10.0,
    weight_decay: float = 0.01,
    text_lr_multiplier: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Optimizer groups: no decay on norms/biases/embeddings, hotter language row.

    A single new embedding row surrounded by an otherwise-converged model learns
    far too slowly at the body's learning rate, hence the multiplier.  With
    ``text_lr_multiplier`` the (row-masked) text embedding gets its own group for
    the same reason: its rows start near zero and must grow to the norm of
    trained tokens.  Embedding groups never decay - decoupled weight decay would
    shrink the masked rows even though their gradient is zero.
    """
    lang, text, no_decay, decay = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("lang_embedding"):
            lang.append(param)
        elif text_lr_multiplier is not None and name == "text_embedding.weight":
            text.append(param)
        elif param.ndim == 1 or name.endswith(".bias") or "embedding" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    groups: List[Dict[str, object]] = []
    if decay:
        groups.append({"params": decay, "lr": base_lr, "weight_decay": weight_decay, "name": "decay"})
    if no_decay:
        groups.append({"params": no_decay, "lr": base_lr, "weight_decay": 0.0, "name": "no_decay"})
    if lang:
        groups.append({
            "params": lang,
            "lr": base_lr * lang_lr_multiplier,
            "weight_decay": 0.0,
            "name": "lang_embedding",
        })
    if text:
        groups.append({
            "params": text,
            "lr": base_lr * text_lr_multiplier,
            "weight_decay": 0.0,
            "name": "text_embedding",
        })
    return groups


def describe_trainable(model, limit: int = 24) -> str:
    rows = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    rows.sort(key=lambda kv: -kv[1])
    total = sum(n for _, n in rows)
    all_params = sum(p.numel() for p in model.parameters())
    lines = [
        "trainable: {:,} / {:,} params ({:.2f}%)".format(
            total, all_params, 100.0 * total / max(1, all_params)
        )
    ]
    for name, count in rows[:limit]:
        lines.append("  {:<52} {:>12,}".format(name, count))
    if len(rows) > limit:
        lines.append("  ... and " + str(len(rows) - limit) + " more tensors")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #

@torch.no_grad()
def export_inference_checkpoint(
    model,
    path: str | Path,
    merge_adapters: bool = True,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Write a ``gpt.pth`` that stock ``infer_v2_5.py`` can load unmodified."""
    merged = merge_lora(model) if merge_adapters and has_lora(model) else 0

    state = {k: v.detach().to(torch.float32).cpu() for k, v in model.state_dict().items()}
    state = {k: v for k, v in state.items() if not k.startswith("inference_model.")}
    state = {k: v for k, v in state.items() if ".lora_" not in k}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {"model": state}
    if metadata:
        payload["itts25ft"] = metadata
    torch.save(payload, path)
    return {"merged_adapters": merged, "tensors": len(state), "path": str(path)}
