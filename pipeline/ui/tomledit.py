"""pipeline.toml'da tek tek değer değiştirir; yorumlar ve düzen korunur."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

_SECTION = re.compile(r"^\s*\[([^\[\]]+)\]\s*(#.*)?$")
_KEY = re.compile(r"^(\s*)([A-Za-z0-9_]+)(\s*=\s*)(.*)$")


def toml_value(value: Any) -> str:
    """Python değerini TOML yazımına çevirir (dizgeler tek tırnakla: Windows yolları kaçışsız kalır)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        if "'" in value or "\n" in value:
            return json.dumps(value, ensure_ascii=False)
        return "'" + value + "'"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    raise TypeError("Desteklenmeyen değer: {!r}".format(value))


def _split_comment(rest: str) -> Tuple[str, str]:
    """'değer   # yorum' -> ('değer', '   # yorum'); tırnak içindeki # yorum sayılmaz."""
    quote = None
    for i, ch in enumerate(rest):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#":
            j = i
            while j > 0 and rest[j - 1] in " \t":
                j -= 1
            return rest[:j], rest[j:]
    return rest.rstrip(), ""


def set_values(text: str, updates: Dict[str, Dict[str, Any]]) -> str:
    """updates = {"bölüm": {"anahtar": değer}}. Var olan anahtarı değiştirir, yoksa bölümün sonuna ekler."""
    lines = text.splitlines()
    pending = {sec: dict(vals) for sec, vals in updates.items()}
    section = None
    last_line_of = {}
    out = []
    for line in lines:
        m = _SECTION.match(line)
        if m:
            section = m.group(1).strip()
            out.append(line)
            last_line_of[section] = len(out) - 1
            continue
        k = _KEY.match(line)
        if k and section in pending and k.group(2) in pending[section]:
            value, comment = _split_comment(k.group(4))
            head = k.group(1) + k.group(2) + k.group(3)
            new = head + toml_value(pending[section].pop(k.group(2)))
            if comment:
                # yorum eski sütununda kalsın
                column = len(head) + len(value) + len(comment) - len(comment.lstrip())
                new += " " * max(1, column - len(new)) + comment.lstrip()
            line = new
        out.append(line)
        if section is not None and line.strip() and not line.lstrip().startswith("#"):
            last_line_of[section] = len(out) - 1
    # dosyada olmayan anahtarlar: bölümün son satırından sonra
    for sec, vals in pending.items():
        if not vals:
            continue
        if sec not in last_line_of:
            out += ["", "[" + sec + "]"]
            last_line_of[sec] = len(out) - 1
        at = last_line_of[sec] + 1
        for key, val in vals.items():
            out.insert(at, "{} = {}".format(key, toml_value(val)))
            at += 1
    return "\n".join(out) + "\n"
