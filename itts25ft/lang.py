"""
Language plumbing for IndexTTS-2.5 finetuning.

IndexTTS-2.5 conditions the AR model on language in *two* independent places:

  1. a text-side control token ``<|xx|>`` prepended to every segment, which the
     Whisper tiktoken vocabulary carries as a special token, and
  2. ``UnifiedVoice.lang_embedding[lang_id]``, a single vector added to every
     text position (see ``prepare_gpt_inputs``).

Both must agree, at preprocessing time and at inference time, or the model will
happily generate the wrong language.  This module is the single source of truth.

Facts about the shipped 2.5 checkpoint (verified against the upstream tree):

  * ``LANGUAGES`` holds 106 entries -> ``lang_embedding`` has 107 rows.
  * only the **first 99** entries own a ``<|xx|>`` special token, because
    ``get_encoding()`` is called with ``num_languages=99``.
  * the release was trained on en(0), zh(1), es(3), ja(7), ar(13).  Every other
    row of ``lang_embedding`` still holds its random initialisation - which is
    exactly the row we are going to train.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

# Populated lazily so that `import itts25ft.lang` works before bootstrap().
_LANGUAGE_DICT: Optional[Dict[str, int]] = None
_LANGUAGE_KEYS: Optional[List[str]] = None

#: Languages the public 2.5 checkpoint was actually trained on.  Used for
#: replay-mixing and as candidates for `--lang-init-from`.
PRETRAINED_LANGS = ("zh", "en", "ja", "es", "ar")

#: Only these rows own a ``<|xx|>`` text control token.
NUM_SPECIAL_TOKEN_LANGS = 99


def _ensure_loaded() -> None:
    global _LANGUAGE_DICT, _LANGUAGE_KEYS
    if _LANGUAGE_DICT is not None:
        return
    from indextts.utils.tokenizer import LANGUAGE_DICT  # noqa: WPS433 (late import by design)

    _LANGUAGE_DICT = dict(LANGUAGE_DICT)
    _LANGUAGE_KEYS = list(LANGUAGE_DICT.keys())


def language_dict() -> Dict[str, int]:
    _ensure_loaded()
    return dict(_LANGUAGE_DICT)  # type: ignore[arg-type]


def num_lang_rows() -> int:
    """Row count of ``UnifiedVoice.lang_embedding`` (``len(LANGUAGE_DICT) + 1``)."""
    _ensure_loaded()
    return len(_LANGUAGE_DICT) + 1  # type: ignore[arg-type]


def has_control_token(code: str) -> bool:
    _ensure_loaded()
    idx = _LANGUAGE_DICT.get(code.lower())  # type: ignore[union-attr]
    return idx is not None and idx < NUM_SPECIAL_TOKEN_LANGS


@dataclass(frozen=True)
class LanguageSlot:
    """A resolved (control token, embedding row) pair used end to end."""

    code: str            # what the user calls it, e.g. "tr"
    slot: str            # the vocabulary slot actually used, e.g. "tr" or an alias
    lang_id: int         # row index into lang_embedding
    aliased: bool = False

    @property
    def prefix(self) -> str:
        """Exactly what ``infer_v2_5`` prepends before tokenising."""
        return f"<|{self.slot}|> "

    def describe(self) -> str:
        if self.aliased:
            return (
                f"{self.code!r} -> borrowed slot <|{self.slot}|> (lang_id={self.lang_id}). "
                f"Pass --lang {self.code} --lang-alias {self.slot} at inference too."
            )
        return f"{self.code!r} -> <|{self.slot}|> (lang_id={self.lang_id})"


def resolve(code: str, alias: Optional[str] = None) -> LanguageSlot:
    """
    Map a language code onto a usable (control token, embedding row) slot.

    ``alias`` lets you train a language that is not in Whisper's list by
    borrowing an unused-but-tokenisable slot (e.g. ``--lang az --lang-alias tt``).
    We deliberately do *not* grow the tiktoken vocabulary: new special tokens
    would land past ``number_text_tokens`` and desync every downstream index.
    """
    _ensure_loaded()
    code = code.lower().strip()
    table = _LANGUAGE_DICT  # type: ignore[assignment]

    if alias:
        alias = alias.lower().strip()
        if alias not in table:
            raise ValueError(f"--lang-alias {alias!r} is not a known language slot.")
        if not has_control_token(alias):
            raise ValueError(
                f"--lang-alias {alias!r} has no <|{alias}|> control token "
                f"(only the first {NUM_SPECIAL_TOKEN_LANGS} languages do)."
            )
        if alias in PRETRAINED_LANGS:
            raise ValueError(
                f"Refusing to alias onto {alias!r}: it is one of the languages the base "
                f"model was trained on, and training would overwrite it."
            )
        return LanguageSlot(code=code, slot=alias, lang_id=table[alias], aliased=alias != code)

    if code not in table:
        raise ValueError(
            f"Unknown language {code!r}. Either use a Whisper language code, or pick an "
            f"unused slot with --lang-alias (see itts25ft.lang.free_slots())."
        )
    if not has_control_token(code):
        raise ValueError(
            f"{code!r} exists in LANGUAGE_DICT but has no <|{code}|> control token. "
            f"Re-run with --lang-alias <slot> to borrow one."
        )
    return LanguageSlot(code=code, slot=code, lang_id=table[code])


def free_slots(limit: int = 40) -> List[str]:
    """Tokenisable language slots that the base model never trained."""
    _ensure_loaded()
    out = [
        code
        for code, idx in _LANGUAGE_DICT.items()  # type: ignore[union-attr]
        if idx < NUM_SPECIAL_TOKEN_LANGS and code not in PRETRAINED_LANGS
    ]
    return out[:limit]
