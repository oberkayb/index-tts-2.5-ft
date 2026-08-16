"""
Datasets, batching and language mixing.

Training samples are *pairs*: the speaker prompt and the synthesis target are
different utterances.  This is not an optimisation, it is what stops the model
from learning to copy the prompt's content instead of cloning its timbre - the
"prompt audio bleeding" failure mode.  IndexTTS-2 trains this way and 2.5's
CAMPPlus speaker interface makes it even more important, because a single 192-d
vector is easy to overfit to a handful of new-language speakers.

Two mixing knobs matter for keeping the base model's abilities:

  * **replay** - every manifest carries a language and a sampling weight, so a
    Turkish run can keep, say, 20% of its batches on English and Chinese data
    and never drift away from them.
  * **cross-lingual pairs** - when a speaker has recordings in more than one
    language, ``build_pairs`` can pair a prompt in language A with a target in
    language B.  That is direct supervision for the exact behaviour we want to
    preserve; without such speakers the defence is replay plus low-rank updates.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .extractors import load_features
from .lang import LanguageSlot


# --------------------------------------------------------------------------- #
# Manifest records
# --------------------------------------------------------------------------- #

@dataclass
class ManifestSpec:
    """One manifest plus how often to draw from it."""

    path: Path
    lang: Optional[str] = None       # overrides the record's own "lang"
    alias: Optional[str] = None      # borrowed vocabulary slot, see lang.resolve
    weight: float = 1.0

    @classmethod
    def parse(cls, raw: str) -> "ManifestSpec":
        """Parse ``path[::lang[:alias]][@weight]``.

        Examples::

            data/tr_train.jsonl
            data/tr_train.jsonl::tr
            data/en_replay.jsonl::en@0.25
            data/az_train.jsonl::az:tt@1.0
        """
        value = raw.strip()
        weight = 1.0
        if "@" in value:
            value, weight_str = value.rsplit("@", 1)
            weight = float(weight_str)

        lang = alias = None
        if "::" in value:
            value, lang_part = value.split("::", 1)
            if ":" in lang_part:
                lang, alias = lang_part.split(":", 1)
            else:
                lang = lang_part
        return cls(path=Path(value.strip()).expanduser(), lang=lang, alias=alias, weight=weight)


@dataclass
class PairRecord:
    uid: str
    target_features: Path
    prompt_features: Path
    lang: str
    lang_id: int
    speaker: str = ""
    text_len: int = 0
    code_len: int = 0
    cross_lingual: bool = False
    weight: float = 1.0


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

@dataclass
class DatasetConfig:
    #: Where the emotion vector comes from.  ``target`` matches how IndexTTS-2
    #: disentangles emotion from timbre (emotion reference = the utterance being
    #: predicted); ``prompt`` matches the inference default where the emotion
    #: clip and the speaker clip are the same file.  Training on ``target`` with
    #: some ``prompt_emo_prob`` gives you both behaviours.
    emo_source: str = "target"
    prompt_emo_prob: float = 0.25
    max_text_tokens: int = 600
    max_code_tokens: int = 1815
    min_code_tokens: int = 8


class PairedDataset(Dataset):
    """Prompt/target pairs drawn from one or more language manifests."""

    def __init__(
        self,
        specs: Sequence[ManifestSpec],
        config: Optional[DatasetConfig] = None,
        resolve_slot=None,
        verbose: bool = True,
    ) -> None:
        from . import lang as lang_mod

        self.config = config or DatasetConfig()
        self._resolve = resolve_slot or lang_mod.resolve
        self.records: List[PairRecord] = []
        self.summary: Dict[str, Dict[str, object]] = {}
        self._slots: Dict[str, LanguageSlot] = {}

        for spec in specs:
            self._load(spec, verbose=verbose)

        if not self.records:
            raise RuntimeError(
                "No usable records. Check the manifests and the length limits "
                "(max_text_tokens / max_code_tokens)."
            )

    # -- loading ------------------------------------------------------------ #

    def _slot_for(self, code: str, alias: Optional[str]) -> LanguageSlot:
        key = code + "|" + (alias or "")
        if key not in self._slots:
            self._slots[key] = self._resolve(code, alias)
        return self._slots[key]

    def _load(self, spec: ManifestSpec, verbose: bool) -> None:
        if not spec.path.is_file():
            raise FileNotFoundError("Manifest not found: " + str(spec.path))
        base = spec.path.parent

        kept = skipped_len = skipped_missing = 0
        langs: Dict[str, int] = {}

        with spec.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                code = (spec.lang or rec.get("lang") or "").lower()
                if not code:
                    raise ValueError(
                        "Record " + str(rec.get("id")) + " in " + str(spec.path)
                        + " has no language; set it in the manifest or append ::lang to the path."
                    )
                slot = self._slot_for(code, spec.alias or rec.get("lang_alias"))

                text_len = int(rec.get("text_len", 0))
                code_len = int(rec.get("code_len", 0))
                if text_len > self.config.max_text_tokens or code_len > self.config.max_code_tokens:
                    skipped_len += 1
                    continue
                if code_len < self.config.min_code_tokens:
                    skipped_len += 1
                    continue

                target = self._resolve_path(base, rec["target_features"])
                prompt = self._resolve_path(base, rec.get("prompt_features", rec["target_features"]))
                if not target.is_file() or not prompt.is_file():
                    skipped_missing += 1
                    continue

                self.records.append(
                    PairRecord(
                        uid=str(rec.get("id", target.stem)),
                        target_features=target,
                        prompt_features=prompt,
                        lang=code,
                        lang_id=slot.lang_id,
                        speaker=str(rec.get("speaker", "")),
                        text_len=text_len,
                        code_len=code_len,
                        cross_lingual=bool(rec.get("cross_lingual", False)),
                        weight=spec.weight,
                    )
                )
                kept += 1
                langs[code] = langs.get(code, 0) + 1

        self.summary[str(spec.path)] = {
            "kept": kept,
            "skipped_length": skipped_len,
            "skipped_missing": skipped_missing,
            "weight": spec.weight,
            "languages": langs,
        }
        if verbose:
            print(
                "[data] " + str(spec.path) + ": kept " + str(kept)
                + ", dropped " + str(skipped_len) + " (length) / "
                + str(skipped_missing) + " (missing files), weight=" + str(spec.weight)
            )

    @staticmethod
    def _resolve_path(base: Path, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (base / path)

    # -- Dataset protocol --------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        rec = self.records[index]
        target = load_features(rec.target_features)
        prompt = (
            target
            if rec.prompt_features == rec.target_features
            else load_features(rec.prompt_features)
        )

        emo_from_prompt = self.config.emo_source == "prompt" or (
            self.config.emo_source == "target"
            and random.random() < self.config.prompt_emo_prob
        )
        emo_vec = prompt["emo_vec"] if emo_from_prompt else target["emo_vec"]

        return {
            "id": rec.uid,
            "text_ids": torch.from_numpy(target["text_ids"].astype(np.int64)),
            "codes": torch.from_numpy(target["codes"].astype(np.int64)),
            "spk_emb": torch.from_numpy(prompt["spk_emb"].astype(np.float32)),
            "emo_vec": torch.from_numpy(emo_vec.astype(np.float32)),
            "lang_id": rec.lang_id,
            "lang": rec.lang,
            "cross_lingual": rec.cross_lingual,
        }

    # -- helpers ------------------------------------------------------------ #

    def sample_weights(self) -> np.ndarray:
        return np.asarray([rec.weight for rec in self.records], dtype=np.float64)

    def code_lengths(self) -> np.ndarray:
        return np.asarray([rec.code_len for rec in self.records], dtype=np.int64)

    def language_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for rec in self.records:
            counts[rec.lang] = counts.get(rec.lang, 0) + 1
        return counts


# --------------------------------------------------------------------------- #
# Collation
# --------------------------------------------------------------------------- #

def collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    text_ids = [item["text_ids"] for item in batch]
    codes = [item["codes"] for item in batch]

    return {
        "ids": [item["id"] for item in batch],
        # Padding value 0 is overwritten by set_text_padding / set_mel_padding,
        # which replace everything past the true length with the stop token.
        "text_ids": pad_sequence(text_ids, batch_first=True, padding_value=0),
        "codes": pad_sequence(codes, batch_first=True, padding_value=0),
        "spk_emb": torch.stack([item["spk_emb"] for item in batch]),
        "emo_vec": torch.stack([item["emo_vec"] for item in batch]),
        "lang_ids": torch.tensor([item["lang_id"] for item in batch], dtype=torch.long),
        "text_lengths": torch.tensor([len(t) for t in text_ids], dtype=torch.long),
        "code_lengths": torch.tensor([len(c) for c in codes], dtype=torch.long),
        "langs": [item["lang"] for item in batch],
        "cross_lingual": torch.tensor(
            [bool(item["cross_lingual"]) for item in batch], dtype=torch.bool
        ),
    }


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #

class LengthBucketBatchSampler(torch.utils.data.Sampler):
    """Weighted language mixing + length bucketing.

    Manifest weights decide how often each corpus appears (that is the replay
    mechanism); bucketing then groups similar-length samples so padding does not
    dominate a batch, which matters a lot when semantic sequences run to ~1800
    tokens.
    """

    def __init__(
        self,
        dataset: PairedDataset,
        batch_size: int,
        bucket_multiplier: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 1234,
        use_weights: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_multiplier = max(1, bucket_multiplier)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.lengths = dataset.code_lengths()

        weights = dataset.sample_weights() if use_weights else None
        if weights is not None and not np.allclose(weights, weights[0]):
            self.weights = weights / weights.sum()
        else:
            self.weights = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        n = len(self.dataset)

        if self.weights is not None:
            indices = rng.choice(n, size=n, replace=True, p=self.weights)
        else:
            indices = rng.permutation(n) if self.shuffle else np.arange(n)

        batches: List[List[int]] = []
        chunk = self.batch_size * self.bucket_multiplier
        for start in range(0, len(indices), chunk):
            window = indices[start : start + chunk]
            window = window[np.argsort(self.lengths[window], kind="stable")]
            for offset in range(0, len(window), self.batch_size):
                batch = window[offset : offset + self.batch_size].tolist()
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
