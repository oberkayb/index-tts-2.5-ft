"""Finetuning toolkit for IndexTTS-2.5: add a language, keep the rest."""

__version__ = "0.1.0"

from . import env  # noqa: F401  (bootstrap helper, import has no side effects)

__all__ = ["env", "lang", "textfront", "extractors", "modeling", "losses", "data", "utils"]
