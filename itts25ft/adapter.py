"""
Language adapters: keep a fine-tuned language apart from the stock IndexTTS-2.5 weights.

An adapter holds only what the fine-tune changed:

  * ``lora``     low-rank deltas for the GPT block projections (delta = A @ B),
  * ``rows``     the text-embedding rows of the new language and its language row,
  * ``replace``  tensors trained outright (``mel_head``).

``text_head`` is left out: inference returns latents (``return_latent=True``) and
picks speech codes with ``mel_head`` only, so ``text_head`` never runs.

:class:`LanguageAdapter` attaches to a loaded ``UnifiedVoice`` and switches on or
off per call. Off means every touched tensor holds its stock value again, so the
other languages produce exactly what stock IndexTTS-2.5 produces.

Adapters can be extracted from a trainer checkpoint (``best.pt``) or, when only a
merged ``gpt.pth`` exists, from its difference to the stock weights: a LoRA delta
has rank ``r``, so a truncated SVD recovers it up to float rounding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn

FORMAT = "itts25ft-adapter"
VERSION = 1
LORA_TARGETS = (".attn.c_attn.", ".attn.c_proj.", ".mlp.c_fc.", ".mlp.c_proj.")
ROW_TABLES = ("text_embedding.weight", "lang_embedding.weight")
REPLACE_PREFIXES = ("mel_head.",)
SKIP_PREFIXES = ("text_head.", "inference_model.")


# --------------------------------------------------------------------------- #
# file format (safetensors: no code runs on load)
# --------------------------------------------------------------------------- #

def save_adapter(path: str | Path, lora: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
                 rows: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
                 replace: Mapping[str, torch.Tensor], meta: Mapping[str, object],
                 dtype: torch.dtype = torch.float16) -> Path:
    from safetensors.torch import save_file

    flat: Dict[str, torch.Tensor] = {}
    for name, (a, b) in lora.items():
        flat["lora." + name + ".A"] = a.detach().to(dtype).contiguous().cpu()
        flat["lora." + name + ".B"] = b.detach().to(dtype).contiguous().cpu()
    for name, (index, values) in rows.items():
        flat["rows." + name + ".index"] = index.detach().to(torch.int64).contiguous().cpu()
        flat["rows." + name + ".values"] = values.detach().to(dtype).contiguous().cpu()
    for name, value in replace.items():
        flat["replace." + name] = value.detach().to(dtype).contiguous().cpu()
    header = {"format": FORMAT, "version": str(VERSION), "meta": json.dumps(dict(meta), ensure_ascii=False)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(flat, str(path), metadata=header)
    return path


def load_adapter(path: str | Path):
    """Returns (lora, rows, replace, meta)."""
    from safetensors import safe_open

    lora: Dict[str, Dict[str, torch.Tensor]] = {}
    rows: Dict[str, Dict[str, torch.Tensor]] = {}
    replace: Dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as f:
        header = f.metadata() or {}
        if header.get("format") != FORMAT:
            raise ValueError("Not an itts25ft adapter: " + str(path))
        for key in f.keys():
            kind, rest = key.split(".", 1)
            if kind == "lora":
                name, part = rest.rsplit(".", 1)
                lora.setdefault(name, {})[part] = f.get_tensor(key)
            elif kind == "rows":
                name, part = rest.rsplit(".", 1)
                rows.setdefault(name, {})[part] = f.get_tensor(key)
            elif kind == "replace":
                replace[rest] = f.get_tensor(key)
    meta = json.loads(header.get("meta", "{}"))
    return ({k: (v["A"], v["B"]) for k, v in lora.items()},
            {k: (v["index"], v["values"]) for k, v in rows.items()},
            replace, meta)


def adapter_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / 1048576


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #

def _state_dict(path: str | Path) -> Dict[str, torch.Tensor]:
    state = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    return state


def _changed_rows(base: torch.Tensor, tuned: torch.Tensor) -> torch.Tensor:
    return ((tuned.float() - base.float()).abs().amax(dim=1) > 0).nonzero().flatten()


def _low_rank(delta: torch.Tensor, rank: Optional[int], device: str) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
    """delta [in, out] -> A [in, r], B [r, out]. rank=None picks the knee of the spectrum."""
    d = delta.float().to(device)
    u, s, vh = torch.linalg.svd(d, full_matrices=False)
    energy = s.pow(2)
    if rank is None:
        tail = 1 - energy.cumsum(0) / energy.sum()
        below = (tail < 1e-6).nonzero()
        rank = int(below[0].item()) + 1 if len(below) else min(256, len(s))
    root = s[:rank].sqrt()
    a = u[:, :rank] * root
    b = root[:, None] * vh[:rank]
    err = ((a @ b - d).norm() / d.norm().clamp_min(1e-12)).item()
    return a.cpu(), b.cpu(), rank, err


def extract_from_merged(base_path: str | Path, tuned_path: str | Path, rank: Optional[int] = None,
                        device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Adapter from a merged (exported) gpt.pth and the stock gpt.pth."""
    base, tuned = _state_dict(base_path), _state_dict(tuned_path)
    lora, rows, replace, report = {}, {}, {}, {"ranks": set(), "worst_rel_error": 0.0, "unexpected": []}
    for key in sorted(set(base) & set(tuned)):
        if key.startswith(SKIP_PREFIXES) or not base[key].is_floating_point():
            continue
        if torch.equal(base[key], tuned[key]):
            continue
        if any(t in key for t in LORA_TARGETS) and key.endswith(".weight"):
            a, b, r, err = _low_rank(tuned[key].float() - base[key].float(), rank, device)
            lora[key[: -len(".weight")]] = (a, b)
            report["ranks"].add(r)
            report["worst_rel_error"] = max(report["worst_rel_error"], err)
        elif key in ROW_TABLES:
            index = _changed_rows(base[key], tuned[key])
            rows[key] = (index, tuned[key][index])
        elif key.startswith(REPLACE_PREFIXES):
            replace[key] = tuned[key]
        else:
            report["unexpected"].append(key)
    report["ranks"] = sorted(report["ranks"])
    return lora, rows, replace, report


def extract_from_trainer(base_path: str | Path, checkpoint_path: str | Path):
    """Adapter from a trainer checkpoint (``best.pt`` / ``step*.pt``), exact LoRA factors."""
    base = _state_dict(base_path)
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False, mmap=True)
    trained: Dict[str, torch.Tensor] = state["trainable"]
    spec = dict(state.get("spec", {}))
    scaling = float(spec.get("lora_alpha", 64.0)) / float(spec.get("lora_rank", 32))
    lora, rows, replace, report = {}, {}, {}, {"ranks": set(), "worst_rel_error": 0.0, "unexpected": []}
    for key, value in trained.items():
        if key.endswith(".lora_A"):
            name = key[: -len(".lora_A")]
            b = trained[name + ".lora_B"]
            lora[name] = (value.float(), b.float() * scaling)
            report["ranks"].add(value.shape[1])
        elif key.endswith(".lora_B"):
            continue
        elif key in ROW_TABLES:
            index = _changed_rows(base[key], value)
            rows[key] = (index, value[index])
        elif key.startswith(REPLACE_PREFIXES):
            replace[key] = value
        elif not key.startswith(SKIP_PREFIXES):
            report["unexpected"].append(key)
    report["ranks"] = sorted(report["ranks"])
    return lora, rows, replace, report


# --------------------------------------------------------------------------- #
# runtime
# --------------------------------------------------------------------------- #

class SwitchableLoRA(nn.Module):
    """Wraps a HuggingFace ``Conv1D`` (weight [in, out]); adds x @ A @ B while active."""

    def __init__(self, base: nn.Module, a: torch.Tensor, b: torch.Tensor) -> None:
        super().__init__()
        self.base = base
        dtype, device = base.weight.dtype, base.weight.device
        self.register_buffer("lora_A", a.to(device=device, dtype=dtype), persistent=False)
        self.register_buffer("lora_B", b.to(device=device, dtype=dtype), persistent=False)
        self.active = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.active:
            out = out + (x @ self.lora_A.to(x.dtype)) @ self.lora_B.to(x.dtype)
        return out


def _resolve(root: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    parent = root
    parts = dotted.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


class LanguageAdapter:
    """Attach an adapter to a loaded ``UnifiedVoice`` (``IndexTTS2.gpt``) and switch it per call.

        adapter = LanguageAdapter.load("adapters/tr.safetensors").attach(tts.gpt)
        adapter.set_active(lang == adapter.lang)
    """

    def __init__(self, lora, rows, replace, meta) -> None:
        self.lora, self.rows, self.replace, self.meta = lora, rows, replace, dict(meta)
        self.lang = str(self.meta.get("lang", "tr"))
        self.model: Optional[nn.Module] = None
        self.wrappers: list[SwitchableLoRA] = []
        self._stock: Dict[str, torch.Tensor] = {}
        self._tuned: Dict[str, torch.Tensor] = {}
        self.active = False

    @classmethod
    def load(cls, path: str | Path) -> "LanguageAdapter":
        adapter = cls(*load_adapter(path))
        adapter.meta.setdefault("file", str(path))
        return adapter

    @torch.no_grad()
    def attach(self, model: nn.Module) -> "LanguageAdapter":
        if self.model is not None:
            raise RuntimeError("Adapter is already attached.")
        self.model = model
        for name, (a, b) in self.lora.items():
            parent, child_name = _resolve(model, name)
            child = getattr(parent, child_name)
            if isinstance(child, SwitchableLoRA):
                raise RuntimeError("Another adapter is attached at " + name + "; detach it first.")
            wrapper = SwitchableLoRA(child, a, b)
            setattr(parent, child_name, wrapper)
            self.wrappers.append(wrapper)
        state = dict(model.named_parameters())
        for name, (index, values) in self.rows.items():
            weight = state[name]
            idx = index.to(weight.device)
            self._stock[name] = weight.data[idx].clone()
            self._tuned[name] = values.to(device=weight.device, dtype=weight.dtype)
        for name, value in self.replace.items():
            param = state[name]
            self._stock[name] = param.data
            self._tuned[name] = value.to(device=param.device, dtype=param.dtype)
        return self

    @torch.no_grad()
    def set_active(self, on: bool) -> None:
        if self.model is None or on == self.active:
            return
        source = self._tuned if on else self._stock
        state = dict(self.model.named_parameters())
        for name, (index, _) in self.rows.items():
            weight = state[name]
            weight.data.index_copy_(0, index.to(weight.device), source[name])
        for name in self.replace:
            state[name].data = source[name]
        for wrapper in self.wrappers:
            wrapper.active = on
        self.active = on

    @torch.no_grad()
    def detach(self) -> None:
        if self.model is None:
            return
        self.set_active(False)
        for name in self.lora:
            parent, child_name = _resolve(self.model, name)
            wrapper = getattr(parent, child_name)
            if isinstance(wrapper, SwitchableLoRA):
                setattr(parent, child_name, wrapper.base)
        self.model, self.wrappers = None, []
        self._stock.clear()
        self._tuned.clear()

    def describe(self) -> str:
        n_rows = {k.split(".")[0]: int(v[0].numel()) for k, v in self.rows.items()}
        ranks = sorted({int(a.shape[1]) for a, _ in self.lora.values()})
        return "lang={} · {} LoRA (rank {}) · {} · {}".format(
            self.lang, len(self.lora), "/".join(map(str, ranks)) or "-",
            ", ".join("{} {} satır".format(k, v) for k, v in n_rows.items()) or "satır yok",
            ", ".join(k for k in self.replace) or "değiştirilen tablo yok")


def list_adapters(folders: Iterable[Path]) -> list[Path]:
    found = []
    for folder in folders:
        if folder.is_dir():
            found += sorted(folder.glob("*.safetensors"))
    return found
