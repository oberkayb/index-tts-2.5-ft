"""
The IndexTTS-2.5 training forward pass.

Upstream ``UnifiedVoice.forward`` is a *latent extraction* path: it returns
``return_latent=True`` hidden states for the s2mel stage and, in campplus mode,
never applies ``lang_embedding``.  Training the AR model needs the other half -
text and semantic cross-entropy - and it must add the language embedding exactly
where ``prepare_gpt_inputs`` adds it at inference, or the finetuned model will
be conditioned differently than it is sampled.

The sequence assembled here is the same one the model generates from:

    [ spk_emb_proj(campplus) + emo_vec | 0 | 0 ][ text tokens ][ semantic codes ]
      <------------ 3 cond positions ----------->

with ``lang_embedding[lang_id]`` broadcast over every text position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class LossConfig:
    text_weight: float = 0.2
    mel_weight: float = 1.0
    label_smoothing: float = 0.0
    #: Gaussian noise added to the CAMPPlus embedding during training.  The
    #: speaker interface is the only thing carrying timbre across languages;
    #: jittering it stops the model from memorising the handful of speakers in a
    #: small new-language corpus and keeps cloning general.
    spk_noise_std: float = 0.0
    #: Probability of replacing the emotion vector with zeros, so the model keeps
    #: producing sane speech when emotion conditioning is neutral.
    emo_dropout: float = 0.0


def build_conditioning(
    model,
    spk_emb: torch.Tensor,
    emo_vec: torch.Tensor,
    spk_noise_std: float = 0.0,
    emo_dropout: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    """Assemble the 3 conditioning positions the 2.5 GPT expects.

    ``spk_emb`` is the *raw* 192-d CAMPPlus vector; projection happens here, the
    same way ``inference_speech`` does it in campplus mode.
    """
    if training and spk_noise_std > 0:
        spk_emb = spk_emb + torch.randn_like(spk_emb) * spk_noise_std

    latent = model.spk_emb_proj(spk_emb)
    if latent.ndim == 2:
        latent = latent.unsqueeze(1)                      # [B, 1, D]

    if training and emo_dropout > 0:
        keep = (torch.rand(emo_vec.size(0), device=emo_vec.device) >= emo_dropout).to(emo_vec.dtype)
        emo_vec = emo_vec * keep.unsqueeze(-1)

    latent = latent + emo_vec.unsqueeze(1)                # [B, 1, D]
    pad = torch.zeros(
        latent.size(0), 2, latent.size(2), device=latent.device, dtype=latent.dtype
    )
    return torch.cat((latent, pad), dim=1)                # [B, 3, D]


def compute_losses(
    model,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    config: Optional[LossConfig] = None,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Return ``(text_loss, mel_loss, metrics)`` for one batch."""
    config = config or LossConfig()

    spk_emb = batch["spk_emb"].to(device)
    emo_vec = batch["emo_vec"].to(device)
    text_ids = batch["text_ids"].to(device)
    codes = batch["codes"].to(device)
    lang_ids = batch["lang_ids"].to(device)
    text_lengths = batch["text_lengths"].to(device)
    code_lengths = batch["code_lengths"].to(device)

    conds = build_conditioning(
        model,
        spk_emb,
        emo_vec,
        spk_noise_std=config.spk_noise_std,
        emo_dropout=config.emo_dropout,
        training=training,
    )

    # -- text stream ------------------------------------------------------- #
    text_inputs = model.set_text_padding(text_ids.clone(), text_lengths)
    text_inputs = F.pad(text_inputs, (0, 1), value=model.stop_text_token)
    text_inputs, text_targets = model.build_aligned_inputs_and_targets(
        text_inputs, model.start_text_token, model.stop_text_token
    )
    text_emb = model.text_embedding(text_inputs) + model.text_pos_embedding(text_inputs)
    # Same placement as prepare_gpt_inputs: one language vector on every text step.
    text_emb = text_emb + model.lang_embedding(lang_ids).unsqueeze(1)

    # -- semantic stream --------------------------------------------------- #
    mel_inputs = model.set_mel_padding(codes.clone(), code_lengths)
    mel_inputs = F.pad(mel_inputs, (0, 1), value=model.stop_mel_token)
    mel_inputs, mel_targets = model.build_aligned_inputs_and_targets(
        mel_inputs, model.start_mel_token, model.stop_mel_token
    )
    mel_emb = model.mel_embedding(mel_inputs) + model.mel_pos_embedding(mel_inputs)

    text_logits, mel_logits = model.get_logits(
        conds, text_emb, model.text_head, mel_emb, model.mel_head
    )

    # Supervise the real tokens plus the single stop token that follows them.
    text_mask = (
        torch.arange(text_targets.size(1), device=device).unsqueeze(0)
        < (text_lengths + 1).unsqueeze(1)
    )
    mel_mask = (
        torch.arange(mel_targets.size(1), device=device).unsqueeze(0)
        < (code_lengths + 1).unsqueeze(1)
    )

    text_ce = F.cross_entropy(
        text_logits.float(), text_targets, reduction="none",
        label_smoothing=config.label_smoothing,
    )
    mel_ce = F.cross_entropy(
        mel_logits.float(), mel_targets, reduction="none",
        label_smoothing=config.label_smoothing,
    )

    text_loss = (text_ce * text_mask).sum() / text_mask.sum().clamp_min(1)
    mel_loss = (mel_ce * mel_mask).sum() / mel_mask.sum().clamp_min(1)

    with torch.no_grad():
        preds = mel_logits.argmax(dim=1)
        correct = ((preds == mel_targets) & mel_mask).sum()
        metrics = {
            "mel_top1": (correct / mel_mask.sum().clamp_min(1)).item(),
            "mel_ppl": torch.exp(mel_loss.detach().float()).clamp(max=1e6).item(),
            "text_ppl": torch.exp(text_loss.detach().float()).clamp(max=1e6).item(),
        }

    return text_loss, mel_loss, metrics


def total_loss(text_loss: torch.Tensor, mel_loss: torch.Tensor, config: LossConfig) -> torch.Tensor:
    return config.text_weight * text_loss + config.mel_weight * mel_loss
