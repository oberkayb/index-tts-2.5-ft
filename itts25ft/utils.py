"""Small shared helpers: seeding, checkpoint rotation, metric averaging."""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_amp_dtype(name: str) -> Optional[torch.dtype]:
    name = (name or "none").lower()
    if name in ("none", "fp32", "off"):
        return None
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError("Unknown amp dtype: " + repr(name))


class MetricAverager:
    """Running means, weighted by batch size."""

    def __init__(self) -> None:
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, float] = defaultdict(float)

    def update(self, values: Dict[str, float], weight: float = 1.0) -> None:
        for key, value in values.items():
            self._sums[key] += float(value) * weight
            self._counts[key] += weight

    def compute(self) -> Dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums if self._counts[k] > 0}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


@dataclass
class CheckpointManager:
    """Saves training state and keeps only the most recent N snapshots."""

    directory: Path
    keep: int = 3
    prefix: str = "step"
    _history: List[Path] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if self._history is None:
            self._history = []

    def save(self, state: Dict[str, object], step: int) -> Path:
        path = self.directory / (self.prefix + str(step) + ".pt")
        tmp = path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)

        self._history.append(path)
        while len(self._history) > self.keep:
            stale = self._history.pop(0)
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
        return path

    def latest(self) -> Optional[Path]:
        candidates = sorted(
            self.directory.glob(self.prefix + "*.pt"),
            key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0),
        )
        return candidates[-1] if candidates else None


def write_jsonl(path: str | Path, records) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> List[dict]:
    out: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def human_time(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return str(hours) + "h" + str(minutes).zfill(2) + "m"
    if minutes:
        return str(minutes) + "m" + str(secs).zfill(2) + "s"
    return str(secs) + "s"
