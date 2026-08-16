"""
Text frontend that reproduces the IndexTTS-2.5 inference pipeline step for step.

Whatever this module does at preprocessing time is what ``infer_v2_5.py`` must
do at synthesis time, otherwise the finetuned model sees a token distribution it
was never trained on.  The upstream order of operations (infer_v2_5.py, the
"text processing" block) is:

    text = clean_pattern.sub(char_rep_map, text)      # punctuation unification
    text = <language specific normalisation>          # numbers, dates, currency
    text = <language specific casing>                 # lower for zh/ja/en, upper for es
    text = apply_pronunciation_annotations(text)      # <word|PHONES> -> control tokens
    text = re.sub(control-token pattern, upper, text) # uppercase control tokens
    ids  = tokenizer.encode(f'<|{lang}|> ' + text, allowed_special='all')

Note that 2.5 dropped the SentencePiece BPE of IndexTTS-2: the text vocabulary
is now Whisper's ``multilingual_zh_ja_yue_char_del`` tiktoken.  It is byte level,
so any script encodes without vocabulary surgery - the only cost is tokens/char
efficiency, which :meth:`TextFrontend.efficiency_report` measures for you.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .lang import LanguageSlot

# --------------------------------------------------------------------------- #
# Casing helpers
# --------------------------------------------------------------------------- #

def turkish_lower(text: str) -> str:
    """Turkish-correct lowercasing (I -> i-dotless, I-dot -> i)."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def turkish_upper(text: str) -> str:
    return text.replace("i", "İ").replace("ı", "I").upper()


CASE_FUNCS: Dict[str, Callable[[str], str]] = {
    "none": lambda t: t,
    "lower": lambda t: t.lower(),
    "upper": lambda t: t.upper(),
    "tr_lower": turkish_lower,
    "tr_upper": turkish_upper,
}


# --------------------------------------------------------------------------- #
# Turkish text normalisation (reference implementation for a "new" language)
# --------------------------------------------------------------------------- #

_TR_ONES = ["", "bir", "iki", "üç", "dört", "beş",
            "altı", "yedi", "sekiz", "dokuz"]
_TR_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli",
            "altmış", "yetmiş", "seksen", "doksan"]
_TR_SCALES = ["", "bin", "milyon", "milyar", "trilyon", "katrilyon"]

_TR_ORDINAL = {
    1: "birinci", 2: "ikinci", 3: "üçüncü", 4: "dördüncü",
    5: "beşinci", 6: "altıncı", 7: "yedinci", 8: "sekizinci",
    9: "dokuzuncu", 10: "onuncu", 20: "yirminci", 30: "otuzuncu",
    40: "kırkıncı", 50: "ellinci", 60: "altmışıncı",
    70: "yetmişinci", 80: "sekseninci", 90: "doksanıncı",
    100: "yüzüncü", 1000: "bininci",
}

_TR_CURRENCY = {
    "₺": "lira", "tl": "lira", "$": "dolar", "€": "euro", "£": "sterlin",
}

_TR_ABBREV = {
    "vb.": "ve benzeri", "vs.": "vesaire", "örn.": "örneğin",
    "bkz.": "bakınız", "dr.": "doktor", "prof.": "profesör",
    "doç.": "doçent", "av.": "avukat", "sn.": "sayın",
    "yy.": "yüzyıl", "sok.": "sokak", "cad.": "cadde",
    "no.": "numara", "tel.": "telefon", "yak.": "yaklaşık",
}

_TR_UNITS = {
    "km/h": "kilometre bölü saat",
    "km": "kilometre", "cm": "santimetre", "mm": "milimetre", "kg": "kilogram",
    "gr": "gram", "mg": "miligram", "ml": "mililitre", "lt": "litre",
    "m2": "metrekare", "m3": "metreküp",
    "kb": "kilobayt", "mb": "megabayt", "gb": "gigabayt", "tb": "terabayt",
}


def _tr_three_digits(n: int) -> str:
    """0..999 -> Turkish words. 'bir yuz' is wrong; it is just 'yuz'."""
    out: List[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        out.append("yüz" if hundreds == 1 else _TR_ONES[hundreds] + " yüz")
    tens, ones = divmod(rest, 10)
    if tens:
        out.append(_TR_TENS[tens])
    if ones:
        out.append(_TR_ONES[ones])
    return " ".join(out)


def tr_number_to_words(n: int) -> str:
    if n < 0:
        return "eksi " + tr_number_to_words(-n)
    if n == 0:
        return "sıfır"

    groups: List[int] = []
    remaining = n
    while remaining > 0:
        remaining, rem = divmod(remaining, 1000)
        groups.append(rem)
    if len(groups) > len(_TR_SCALES):
        # Absurdly large: read it digit by digit rather than inventing scales.
        return " ".join(_TR_ONES[int(d)] or "sıfır" for d in str(n))

    parts: List[str] = []
    for idx in range(len(groups) - 1, -1, -1):
        value = groups[idx]
        if value == 0:
            continue
        scale = _TR_SCALES[idx]
        # "bir bin" is wrong; a thousands group of exactly 1 is just "bin"
        if idx == 1 and value == 1:
            parts.append("bin")
        else:
            parts.append((_tr_three_digits(value) + " " + scale).strip())
    return " ".join(parts).strip()


def tr_ordinal_to_words(n: int) -> str:
    if n in _TR_ORDINAL:
        return _TR_ORDINAL[n]
    base = tr_number_to_words(n)
    last = n % 10
    if last == 0:
        tail = n % 100
        if tail in _TR_ORDINAL:
            words = base.split()
            return " ".join(words[:-1] + [_TR_ORDINAL[tail]])
        return base + "ıncı"
    words = base.split()
    return " ".join(words[:-1] + [_TR_ORDINAL[last]])


def _tr_digit_string(digits: str) -> str:
    return " ".join(_TR_ONES[int(d)] or "sıfır" for d in digits)


def _tr_num_token(raw: str) -> str:
    """Read a possibly-formatted numeric literal such as '1.234,50'."""
    raw = raw.replace(".", "")
    if "," in raw:
        whole, frac = raw.split(",", 1)
        return tr_number_to_words(int(whole)) + " virgül " + _tr_digit_string(frac)
    return tr_number_to_words(int(raw))


def normalize_turkish(text: str) -> str:
    """Expand numbers, currency, units and common abbreviations into words."""
    text = unicodedata.normalize("NFC", text)

    for abbr, full in _TR_ABBREV.items():
        text = re.sub(r"(?<!\w)" + re.escape(abbr), full, text, flags=re.IGNORECASE)

    # percentages: %50 and 50%
    text = re.sub(r"%\s*(\d[\d.,]*)", lambda m: "yüzde " + _tr_num_token(m.group(1)), text)
    text = re.sub(r"(\d[\d.,]*)\s*%", lambda m: "yüzde " + _tr_num_token(m.group(1)), text)

    # currency: 25TL / 25 lira-sign / $25
    for sym, word in _TR_CURRENCY.items():
        text = re.sub(
            r"(\d[\d.,]*)\s*" + re.escape(sym) + r"(?!\w)",
            lambda m, w=word: _tr_num_token(m.group(1)) + " " + w,
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            re.escape(sym) + r"\s*(\d[\d.,]*)",
            lambda m, w=word: _tr_num_token(m.group(1)) + " " + w,
            text,
            flags=re.IGNORECASE,
        )

    # clock: 14:30 -> "on dort otuz"
    text = re.sub(
        r"\b([01]?\d|2[0-3]):([0-5]\d)\b",
        lambda m: tr_number_to_words(int(m.group(1))) + " " + tr_number_to_words(int(m.group(2))),
        text,
    )

    # units directly attached to a number (longest unit first)
    for unit, word in sorted(_TR_UNITS.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(
            r"(\d)\s*" + re.escape(unit) + r"(?!\w)",
            lambda m, w=word: m.group(1) + " " + w,
            text,
            flags=re.IGNORECASE,
        )

    # ordinals written as "3." at a token boundary
    text = re.sub(r"\b(\d+)\.(?=\s|$)", lambda m: tr_ordinal_to_words(int(m.group(1))), text)

    # thousands separators: 1.234.567 -> 1234567
    text = re.sub(r"\b\d{1,3}(?:\.\d{3})+\b", lambda m: m.group(0).replace(".", ""), text)

    # decimals with comma, then plain integers
    text = re.sub(
        r"\b(\d+),(\d+)\b",
        lambda m: tr_number_to_words(int(m.group(1))) + " virgül " + _tr_digit_string(m.group(2)),
        text,
    )
    text = re.sub(r"\b\d+\b", lambda m: tr_number_to_words(int(m.group(0))), text)

    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Normaliser registry
# --------------------------------------------------------------------------- #

def _passthrough(text: str) -> str:
    return text


def _nemo_normalizer(lang: str) -> Callable[[str], str]:
    from indextts.utils.nemo_tn import normalize_text as nemo_text_normalize

    return lambda text: nemo_text_normalize(text, lang)


def resolve_normalizer(spec: str) -> Callable[[str], str]:
    """
    ``spec`` is one of:
      * ``none``                    - no expansion (do it offline yourself)
      * ``turkish``                 - the reference implementation above
      * ``nemo:<lang>``             - upstream NeMo TN (supports ja, es, ...)
      * ``module.path:function``    - your own ``f(str) -> str``
    """
    spec = (spec or "none").strip()
    if spec in ("none", ""):
        return _passthrough
    if spec in ("turkish", "tr"):
        return normalize_turkish
    if spec.startswith("nemo:"):
        return _nemo_normalizer(spec.split(":", 1)[1])
    if ":" in spec:
        import importlib

        module_name, func_name = spec.rsplit(":", 1)
        module = importlib.import_module(module_name)
        return getattr(module, func_name)
    raise ValueError("Unrecognised normalizer spec: " + repr(spec))


# --------------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------------- #

@dataclass
class TextFrontendConfig:
    normalizer: str = "none"
    case: str = "lower"
    apply_pronunciation: bool = True


class TextFrontend:
    """Turns raw transcripts into the exact id sequence the GPT is trained on."""

    def __init__(
        self,
        model_dir: str | Path,
        slot: LanguageSlot,
        config: Optional[TextFrontendConfig] = None,
    ) -> None:
        from indextts.utils.front import TextNormalizer
        from indextts.utils.tokenizer import get_tokenizer

        self.slot = slot
        self.config = config or TextFrontendConfig()
        self.tokenizer = get_tokenizer(multilingual=True, model_dir=str(model_dir))

        # TextNormalizer.load() pulls in heavyweight zh/en TN models we do not
        # need here; the punctuation table is available without calling it.
        punct = TextNormalizer()
        self._clean_pattern = punct.clean_pattern
        self._char_rep_map = punct.char_rep_map

        self._normalize = resolve_normalizer(self.config.normalizer)
        try:
            self._case = CASE_FUNCS[self.config.case]
        except KeyError as exc:
            raise ValueError(
                "Unknown case mode " + repr(self.config.case)
                + "; pick one of " + str(sorted(CASE_FUNCS))
            ) from exc

    # -- text -> text ------------------------------------------------------- #

    def clean(self, text: str) -> str:
        from indextts.infer_v2_5 import apply_pronunciation_annotations

        text = self._clean_pattern.sub(lambda m: self._char_rep_map[m.group()], text)
        text = self._normalize(text)
        text = self._case(text)
        if self.config.apply_pronunciation:
            text = apply_pronunciation_annotations(text)
        text = re.sub(r"<\|([^|]+)\|>", lambda m: "<|" + m.group(1).upper() + "|>", text)
        return text.strip()

    # -- text -> ids -------------------------------------------------------- #

    def encode(self, text: str, already_clean: bool = False) -> List[int]:
        """Token ids *including* the leading ``<|lang|>`` control token.

        The trailing stop token is intentionally omitted: the trainer appends it
        through ``UnifiedVoice.build_aligned_inputs_and_targets``.
        """
        if not already_clean:
            text = self.clean(text)
        return self.tokenizer.encode(self.slot.prefix + text, allowed_special="all")

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids))

    # -- diagnostics -------------------------------------------------------- #

    def efficiency_report(self, samples: Sequence[str]) -> Dict[str, float]:
        """tokens/char and tokens/word - how well the frozen BPE fits the language.

        Rule of thumb on Whisper's multilingual BPE: <= 0.35 tokens/char is
        comfortable, > 0.5 means the language is being shredded towards bytes and
        you should expect to need more data and more steps.
        """
        total_tokens = total_chars = total_words = 0
        for raw in samples:
            cleaned = self.clean(raw)
            ids = self.tokenizer.encode(cleaned, allowed_special="all")
            total_tokens += len(ids)
            total_chars += len(cleaned)
            total_words += max(1, len(cleaned.split()))
        if total_chars == 0:
            return {"tokens_per_char": 0.0, "tokens_per_word": 0.0, "n_samples": 0.0}
        return {
            "tokens_per_char": total_tokens / total_chars,
            "tokens_per_word": total_tokens / total_words,
            "n_samples": float(len(samples)),
        }

    def max_token_id(self, samples: Sequence[str]) -> int:
        highest = 0
        for raw in samples:
            ids = self.encode(raw)
            if ids:
                highest = max(highest, max(ids))
        return highest
