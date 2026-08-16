"""
Runtime bootstrap: locate the IndexTTS-2.5 source tree + model checkpoints.

This project intentionally lives *outside* the upstream repository so that
`git pull` on index-tts-2.5 never conflicts with the finetuning code.  We only
need the upstream package importable, which this module arranges.

Resolution order for the repo root:
  1. ``--repo`` CLI flag / ``repo`` config key (passed to :func:`bootstrap`)
  2. ``$INDEXTTS25_REPO``
  3. sibling directory ``../index-tts-2.5``
  4. any importable ``indextts`` already on ``sys.path``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

_DEFAULT_SIBLINGS = ("index-tts-2.5", "index-tts2.5", "index-tts")

_bootstrapped: Optional[Path] = None


def _looks_like_repo(path: Path) -> bool:
    return (path / "indextts" / "infer_v2_5.py").is_file()


def find_repo(explicit: Optional[str] = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("INDEXTTS25_REPO")
    if env:
        candidates.append(Path(env).expanduser())
    here = Path(__file__).resolve().parent.parent
    for name in _DEFAULT_SIBLINGS:
        candidates.append(here.parent / name)

    for cand in candidates:
        if cand and _looks_like_repo(cand):
            return cand.resolve()

    raise FileNotFoundError(
        "Could not locate the IndexTTS-2.5 source tree.\n"
        "Set INDEXTTS25_REPO=/path/to/index-tts-2.5 or pass --repo.\n"
        f"Looked at: {[str(c) for c in candidates]}"
    )


def bootstrap(repo: Optional[str] = None) -> Path:
    """Put the upstream package on ``sys.path`` and return the repo root."""
    global _bootstrapped
    root = find_repo(repo)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    _bootstrapped = root
    return root


def find_model_dir(explicit: Optional[str] = None, repo: Optional[Path] = None) -> Path:
    """Locate the downloaded IndexTTS-2.5 weights directory (``checkpoints/``)."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("INDEXTTS25_MODEL_DIR")
    if env:
        candidates.append(Path(env).expanduser())
    if repo is not None:
        candidates.append(Path(repo) / "checkpoints")

    for cand in candidates:
        if cand and (cand / "config.yaml").is_file():
            return cand.resolve()

    raise FileNotFoundError(
        "Could not find the IndexTTS-2.5 checkpoints directory (needs config.yaml).\n"
        "Download it first, e.g.:\n"
        "  hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints\n"
        f"Looked at: {[str(c) for c in candidates]}"
    )


def load_model_config(model_dir: Path):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(Path(model_dir) / "config.yaml"))
    version = cfg.get("version", None)
    if version is not None and float(version) < 2.5:
        raise ValueError(
            f"config.yaml reports version={version}. This project targets IndexTTS-2.5 "
            "(campplus speaker conditioning + language embedding). Point --model-dir at "
            "the IndexTTS-2.5 weights."
        )
    return cfg
