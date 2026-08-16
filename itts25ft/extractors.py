"""
Offline feature extraction for IndexTTS-2.5 GPT finetuning.

Everything the AR model consumes is precomputed once and cached, so training
never touches wav files, w2v-BERT, or the codec again.  Per utterance we cache:

  ``codes``     int32  [M]    semantic tokens from ``EnhancedCodec.quantize``
  ``text_ids``  int32  [T]    tiktoken ids incl. the ``<|lang|>`` control token
  ``spk_emb``   fp32   [192]  raw CAMPPlus embedding (pre ``spk_emb_proj``)
  ``emo_vec``   fp32   [D]    output of ``UnifiedVoice.get_emovec`` (model_dim)

The extraction path mirrors ``IndexTTS2_5.infer_generator`` exactly:

  * audio is resampled to 16 kHz for w2v-BERT and for the CAMPPlus fbank,
  * w2v-BERT hidden state **17** is standardised with ``wav2vec2bert_stats.pt``,
  * CAMPPlus takes 80-bin Kaldi fbank with ``dither=0`` and per-utterance mean
    subtraction,
  * the emotion vector comes from the GPT's own frozen emotion encoder.

Deviating from any of these silently degrades the finetune, which is why the
extractor owns all of it rather than leaving it to callers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio

#: w2v-BERT layer used as the semantic representation (upstream: get_emb).
SEMANTIC_HIDDEN_LAYER = 17

#: CAMPPlus operates on 16 kHz audio.
SPK_SR = 16000

#: Upstream clips reference audio to 15 s before conditioning.
DEFAULT_MAX_REF_SECONDS = 15.0


@dataclass
class ExtractedFeatures:
    codes: np.ndarray        # int32 [M]
    spk_emb: np.ndarray      # float32 [192]
    emo_vec: np.ndarray      # float32 [model_dim]
    duration: float          # seconds
    sample_rate: int


class FeatureExtractor:
    """Loads the frozen 2.5 encoders once and turns wavs into cached tensors."""

    def __init__(
        self,
        model_dir: str | Path,
        cfg,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        gpt: Optional[torch.nn.Module] = None,
    ) -> None:
        from indextts.codec.models import EnhancedCodec
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
        from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

        self.model_dir = Path(model_dir)
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.dtype = dtype

        # -- w2v-BERT 2.0 ---------------------------------------------------- #
        w2v_dir = self.model_dir / "hf_cache" / "w2v-bert-2.0"
        if not w2v_dir.is_dir():
            from indextts.utils.model_download import ensure_models_available

            w2v_dir = Path(ensure_models_available(str(self.model_dir))["w2v_bert"])
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            str(w2v_dir), local_files_only=True
        )
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(
            str(w2v_dir), local_files_only=True
        ).to(self.device).eval()

        stats = torch.load(self.model_dir / cfg.w2v_stat, map_location="cpu")
        self.semantic_mean = stats["mean"].to(self.device)
        self.semantic_std = torch.sqrt(stats["var"]).to(self.device)

        # -- semantic codec -------------------------------------------------- #
        self.semantic_codec = EnhancedCodec(**cfg.semantic_codec, cfg=cfg.semantic_codec)
        self.semantic_codec.load_checkpoint(str(self.model_dir / "codec.pth"))
        self.semantic_codec = self.semantic_codec.to(self.device).eval()

        # -- CAMPPlus speaker encoder ---------------------------------------- #
        campplus_path = self.model_dir / "hf_cache" / "campplus_cn_common.bin"
        if not campplus_path.is_file():
            from indextts.utils.model_download import ensure_models_available

            campplus_path = Path(ensure_models_available(str(self.model_dir))["campplus"])
        campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus.load_state_dict(torch.load(campplus_path, map_location="cpu"))
        self.campplus = campplus.to(self.device).eval()

        # -- GPT (only its frozen emotion branch is used here) ---------------- #
        self.gpt = gpt
        if self.gpt is not None:
            self.gpt = self.gpt.to(self.device).eval()

        self._resamplers: Dict[Tuple[int, int], torchaudio.transforms.Resample] = {}

    # ------------------------------------------------------------------ #
    # audio helpers
    # ------------------------------------------------------------------ #

    def _resample(self, wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
        if src_sr == dst_sr:
            return wav
        key = (src_sr, dst_sr)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(src_sr, dst_sr)
        return self._resamplers[key](wav)

    def load_audio(
        self,
        path: str | Path,
        max_seconds: Optional[float] = None,
    ) -> Tuple[torch.Tensor, int, float]:
        """Mono waveform as [1, N] plus its sample rate and untruncated duration."""
        wav, sr = torchaudio.load(str(path))
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        duration = wav.size(-1) / sr
        if max_seconds is not None and duration > max_seconds:
            wav = wav[:, : int(max_seconds * sr)]
        return wav, sr, duration

    # ------------------------------------------------------------------ #
    # individual features
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def w2v_embedding(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """Standardised w2v-BERT hidden state 17, shape [1, T, 1024]."""
        inputs = self.feature_extractor(wav_16k, sampling_rate=SPK_SR, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        out = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = out.hidden_states[SEMANTIC_HIDDEN_LAYER]
        return (feat - self.semantic_mean) / self.semantic_std

    @torch.no_grad()
    def semantic_codes(self, emb: torch.Tensor) -> torch.Tensor:
        """Semantic token ids, shape [1, M] (codec downsamples w2v frames by 2)."""
        codes, _ = self.semantic_codec.quantize(emb)
        if codes.ndim == 1:
            codes = codes.unsqueeze(0)
        return codes

    @torch.no_grad()
    def speaker_embedding(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """Raw 192-d CAMPPlus embedding, shape [1, 192].

        This is fed to the model *unprojected*; ``UnifiedVoice.spk_emb_proj``
        maps it into model_dim during the forward pass.
        """
        feat = torchaudio.compliance.kaldi.fbank(
            wav_16k.to(self.device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=SPK_SR,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        return self.campplus(feat.unsqueeze(0))

    @torch.no_grad()
    def emotion_vector(self, emb: torch.Tensor) -> torch.Tensor:
        """``get_emovec`` output, shape [1, model_dim].

        Equivalent to ``merge_emovec(x, x, ..., alpha)`` when the emotion prompt
        and the speaker prompt are the same clip, which is the inference default.
        """
        if self.gpt is None:
            raise RuntimeError(
                "FeatureExtractor was built without a GPT; emotion vectors need "
                "UnifiedVoice.get_emovec. Pass gpt=... to the constructor."
            )
        lengths = torch.tensor([emb.shape[1]], device=emb.device)
        return self.gpt.get_emovec(emb, lengths)

    # ------------------------------------------------------------------ #
    # one-shot utterance processing
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def process(
        self,
        audio_path: str | Path,
        max_ref_seconds: float = DEFAULT_MAX_REF_SECONDS,
        max_target_seconds: Optional[float] = None,
    ) -> ExtractedFeatures:
        """Extract every cached feature for one utterance.

        ``max_target_seconds`` bounds the *semantic code* side (the thing the GPT
        must predict); ``max_ref_seconds`` bounds the conditioning side, matching
        the 15 s clip upstream applies to reference audio.
        """
        wav, sr, duration = self.load_audio(audio_path, max_seconds=max_target_seconds)
        wav_16k = self._resample(wav, sr, SPK_SR)

        emb = self.w2v_embedding(wav_16k)
        codes = self.semantic_codes(emb)

        ref_samples = int(max_ref_seconds * SPK_SR)
        wav_ref = wav_16k[:, :ref_samples] if wav_16k.size(-1) > ref_samples else wav_16k
        spk = self.speaker_embedding(wav_ref)

        emb_ref = emb if wav_ref.size(-1) == wav_16k.size(-1) else self.w2v_embedding(wav_ref)
        emo = self.emotion_vector(emb_ref)

        return ExtractedFeatures(
            codes=codes.squeeze(0).to(torch.int32).cpu().numpy(),
            spk_emb=spk.squeeze(0).float().cpu().numpy(),
            emo_vec=emo.squeeze(0).float().cpu().numpy(),
            duration=duration,
            sample_rate=sr,
        )


def save_features(path: str | Path, features: ExtractedFeatures, text_ids: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        codes=features.codes.astype(np.int32),
        text_ids=np.asarray(text_ids, dtype=np.int32),
        spk_emb=features.spk_emb.astype(np.float32),
        emo_vec=features.emo_vec.astype(np.float32),
    )


def load_features(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(str(path)) as data:
        return {key: data[key] for key in ("codes", "text_ids", "spk_emb", "emo_vec")}
