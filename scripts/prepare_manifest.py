#!/usr/bin/env python3
"""
Step 1: turn a raw corpus into a normalised utterance manifest.

Accepts the layouts people actually have:

  * ``jsonl``    - one object per line with audio/text/speaker keys
  * ``csv``/``tsv`` - delimited, with ``--columns`` naming the fields
  * ``ljspeech`` - ``id|text|normalised_text`` with wavs in ``--audio-dir``
  * ``folders``  - ``<root>/<speaker>/<utt>.wav`` plus a sibling ``.txt``/``.lab``

The output is one JSON object per utterance::

    {"id": ..., "audio": ..., "text": ..., "speaker": ..., "lang": "tr", "duration": ...}

``duration`` is carried over when the source has it (jsonl ``duration`` key, or
a ``duration`` csv column); it enables the quality filters below and lets
``preprocess.py`` skip over-long clips without decoding them.

Quality filters for ASR-transcribed corpora (all off by default):

  * ``--min-duration`` / ``--max-duration``  seconds, from the metadata
  * ``--min-cps`` / ``--max-cps``            transcript characters per second;
    outliers are usually misaligned or hallucinated transcripts
  * ``--max-same-text N``                   keep at most N clips per identical
    transcript (sponsor reads, intros, outros)
  * ``--exclude-regex``                     drop transcripts matching a pattern

Speaker identity matters more than usual here: pairing prompt and target
utterances by speaker is what keeps the model cloning timbre rather than
parroting the prompt.  If your corpus has no speaker labels, everything falls
into one bucket and ``build_pairs.py`` will warn you about it.

Example
-------
    python scripts/prepare_manifest.py \
        --format csv --input data/tr/metadata.csv --columns audio,text,speaker \
        --audio-dir data/tr/wavs --lang tr --output data/tr/utterances.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from itts25ft.utils import write_jsonl  # noqa: E402

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")
TEXT_SUFFIXES = (".txt", ".lab", ".normalized.txt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an utterance manifest from a raw corpus.")
    p.add_argument("--format", required=True,
                   choices=("jsonl", "csv", "tsv", "ljspeech", "folders"))
    p.add_argument("--input", type=Path, default=None,
                   help="Metadata file (all formats except 'folders').")
    p.add_argument("--audio-dir", type=Path, default=None,
                   help="Root for relative audio paths / the tree to scan for 'folders'.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lang", required=True, help="Language code for every utterance, e.g. tr")
    p.add_argument("--columns", default="audio,text,speaker",
                   help="Field order for csv/tsv, or key names for jsonl.")
    p.add_argument("--delimiter", default=None, help="Override the delimiter for csv/tsv.")
    p.add_argument("--has-header", action="store_true", help="csv/tsv only: skip the first row.")
    p.add_argument("--speaker", default=None,
                   help="Constant speaker id when the corpus has no speaker column.")
    p.add_argument("--speaker-from-path", type=int, default=None,
                   help="Derive the speaker from the Nth path component from the right (1 = parent dir).")
    p.add_argument("--min-chars", type=int, default=2, help="Drop transcripts shorter than this.")
    p.add_argument("--absolute-paths", action="store_true",
                   help="Store absolute audio paths instead of paths relative to the manifest.")
    p.add_argument("--min-duration", type=float, default=0.0,
                   help="Drop clips shorter than this (seconds, from the metadata duration).")
    p.add_argument("--max-duration", type=float, default=0.0,
                   help="Drop clips longer than this (seconds, from the metadata duration); 0 = off.")
    p.add_argument("--min-cps", type=float, default=0.0,
                   help="Drop transcripts slower than this many characters per second; 0 = off.")
    p.add_argument("--max-cps", type=float, default=0.0,
                   help="Drop transcripts faster than this many characters per second; 0 = off.")
    p.add_argument("--max-same-text", type=int, default=0,
                   help="Keep at most N clips sharing an identical transcript; 0 = off.")
    p.add_argument("--exclude-regex", action="append", default=[],
                   help="Drop transcripts matching this regex (repeatable, case-insensitive).")
    return p.parse_args()


def _resolve_audio(raw: str, audio_dir: Optional[Path]) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if audio_dir is not None:
        candidate = audio_dir / path
        if candidate.exists():
            return candidate
        if not path.suffix:
            for suffix in AUDIO_SUFFIXES:
                alt = audio_dir / (str(path) + suffix)
                if alt.exists():
                    return alt
        return candidate
    return path


def _speaker_from_path(path: Path, depth: int) -> str:
    parts = path.resolve().parts
    index = len(parts) - 1 - depth
    return parts[index] if 0 <= index < len(parts) else "unknown"


def iter_jsonl(args: argparse.Namespace) -> Iterator[Dict[str, str]]:
    keys = [k.strip() for k in args.columns.split(",")]
    audio_key = keys[0] if keys else "audio"
    text_key = keys[1] if len(keys) > 1 else "text"
    speaker_key = keys[2] if len(keys) > 2 else "speaker"

    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            yield {
                "audio": str(record.get(audio_key, record.get("audio", ""))),
                "text": str(record.get(text_key, record.get("text", ""))),
                "speaker": str(record.get(speaker_key, record.get("speaker", "") or "")),
                "duration": record.get("duration"),
            }


def iter_delimited(args: argparse.Namespace) -> Iterator[Dict[str, str]]:
    delimiter = args.delimiter or ("\t" if args.format == "tsv" else ",")
    columns = [c.strip() for c in args.columns.split(",")]
    with args.input.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        if args.has_header:
            next(reader, None)
        for row in reader:
            if not row:
                continue
            record = {name: (row[i] if i < len(row) else "") for i, name in enumerate(columns)}
            yield {
                "audio": record.get("audio", ""),
                "text": record.get("text", ""),
                "speaker": record.get("speaker", ""),
                "duration": record.get("duration") or None,
            }


def iter_ljspeech(args: argparse.Namespace) -> Iterator[Dict[str, str]]:
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            uid = parts[0]
            text = parts[2] if len(parts) > 2 and parts[2].strip() else parts[1]
            yield {"audio": uid, "text": text, "speaker": ""}


def iter_folders(args: argparse.Namespace) -> Iterator[Dict[str, str]]:
    root = args.audio_dir
    if root is None or not root.is_dir():
        raise SystemExit("--audio-dir must point at an existing directory for --format folders")
    for audio in sorted(root.rglob("*")):
        if audio.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        text = ""
        for suffix in TEXT_SUFFIXES:
            candidate = audio.with_suffix(suffix)
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8").strip()
                break
        if not text:
            continue
        yield {"audio": str(audio), "text": text, "speaker": ""}


READERS = {
    "jsonl": iter_jsonl,
    "csv": iter_delimited,
    "tsv": iter_delimited,
    "ljspeech": iter_ljspeech,
    "folders": iter_folders,
}


def main() -> int:
    args = parse_args()
    if args.format != "folders" and args.input is None:
        raise SystemExit("--input is required for --format " + args.format)

    records: List[Dict[str, object]] = []
    seen_ids: Dict[str, int] = {}
    missing = short = no_duration = 0
    speakers: Dict[str, int] = {}
    dropped = {"duration": 0, "chars/sec": 0, "same text": 0, "regex": 0}
    same_text: Dict[str, int] = {}
    excludes = [re.compile(rx, re.IGNORECASE) for rx in args.exclude_regex]
    needs_duration = bool(args.min_duration or args.max_duration or args.min_cps or args.max_cps)
    out_dir = args.output.resolve().parent

    for row in READERS[args.format](args):
        text = (row.get("text") or "").strip()
        raw_audio = (row.get("audio") or "").strip()
        if not raw_audio:
            continue
        if len(text) < args.min_chars:
            short += 1
            continue
        if any(rx.search(text) for rx in excludes):
            dropped["regex"] += 1
            continue

        try:
            duration = float(row.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        if duration > 0:
            if duration < args.min_duration or (args.max_duration and duration > args.max_duration):
                dropped["duration"] += 1
                continue
            cps = len(text) / duration
            if (args.min_cps and cps < args.min_cps) or (args.max_cps and cps > args.max_cps):
                dropped["chars/sec"] += 1
                continue
        elif needs_duration:
            no_duration += 1

        text_key = " ".join(text.lower().split())
        if args.max_same_text and same_text.get(text_key, 0) >= args.max_same_text:
            dropped["same text"] += 1
            continue

        audio_path = _resolve_audio(raw_audio, args.audio_dir)
        if not audio_path.exists():
            missing += 1
            continue
        same_text[text_key] = same_text.get(text_key, 0) + 1

        speaker = (row.get("speaker") or "").strip()
        if not speaker and args.speaker_from_path is not None:
            speaker = _speaker_from_path(audio_path, args.speaker_from_path)
        if not speaker:
            speaker = args.speaker or "spk0"

        uid = audio_path.stem
        if uid in seen_ids:
            seen_ids[uid] += 1
            uid = uid + "_" + str(seen_ids[uid])
        else:
            seen_ids[uid] = 0

        # preprocess.py resolves relative paths against the manifest's folder,
        # so store them relative to it, not to the current directory.
        stored = str(audio_path.resolve())
        if not args.absolute_paths:
            try:
                stored = os.path.relpath(stored, out_dir)
            except ValueError:  # different drive on Windows: keep it absolute
                pass
        record: Dict[str, object] = {
            "id": uid,
            "audio": stored,
            "text": text,
            "speaker": speaker,
            "lang": args.lang.lower(),
        }
        if duration > 0:
            record["duration"] = round(duration, 3)
        records.append(record)
        speakers[speaker] = speakers.get(speaker, 0) + 1

    if not records:
        raise SystemExit("No usable utterances found - check --input / --audio-dir.")

    count = write_jsonl(args.output, records)

    print("wrote " + str(count) + " utterances -> " + str(args.output))
    print("  speakers        : " + str(len(speakers)))
    print("  missing audio   : " + str(missing))
    print("  too-short text  : " + str(short))
    for reason, n in dropped.items():
        if n:
            print("  dropped ({:<9}): {}".format(reason, n))
    hours = sum(float(r.get("duration", 0.0)) for r in records) / 3600.0
    if hours:
        print("  kept duration   : {:.1f} h".format(hours))
    if no_duration:
        print("  WARNING: " + str(no_duration) + " records had no duration, so the duration and "
              "chars/sec filters could not apply to them.")
    top = sorted(speakers.items(), key=lambda kv: -kv[1])[:8]
    print("  largest speakers: " + ", ".join(s + "(" + str(n) + ")" for s, n in top))
    if len(speakers) == 1:
        print("  NOTE: a single speaker means prompt/target pairs come from one voice; "
              "voice cloning quality on unseen speakers depends almost entirely on the base "
              "model, so keep the learning rate low and prefer --trainable-mode lora.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
