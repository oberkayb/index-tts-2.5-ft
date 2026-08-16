#!/usr/bin/env python3
"""
Step 3: turn utterances into prompt/target training pairs.

The prompt supplies the CAMPPlus speaker vector, the target supplies the text
and the semantic codes to predict.  They must be *different* utterances of the
same speaker: pairing an utterance with itself lets the model read the answer
off the conditioning, which shows up at inference as the prompt's content
leaking into the output.

Cross-lingual pairs
-------------------
If a speaker appears in more than one language, ``--cross-lingual`` emits pairs
whose prompt is in language A and whose target is in language B.  That is the
only *direct* supervision for "clone this voice, speak that language", and it is
the strongest thing you can do to keep 2.5's cross-lingual behaviour while
adding a language.  Without such speakers the defence falls back to replay data
plus low-rank updates, which the trainer handles.

Example
-------
    python scripts/build_pairs.py \
        --manifest data/tr/processed/utterances_train.jsonl \
        --manifest data/en/processed/utterances_train.jsonl \
        --output data/pairs_train.jsonl \
        --pairs-per-target 2 --cross-lingual
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from itts25ft.utils import read_jsonl, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build prompt/target pair manifests.")
    p.add_argument("--manifest", dest="manifests", action="append", type=Path, required=True,
                   help="Utterance manifest from step 2. Repeat for multiple languages.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pairs-per-target", type=int, default=2,
                   help="Prompts sampled per target utterance.")
    p.add_argument("--cross-lingual", action="store_true",
                   help="Also emit pairs whose prompt language differs from the target's.")
    p.add_argument("--cross-lingual-ratio", type=float, default=0.5,
                   help="Share of a multilingual speaker's pairs that cross languages.")
    p.add_argument("--allow-self-pairs", action="store_true",
                   help="Permit prompt == target for speakers with a single utterance.")
    p.add_argument("--max-prompt-code-len", type=int, default=0,
                   help="Prefer prompts no longer than this many semantic tokens (0 = no limit).")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def _feature_path(record: Dict[str, object], manifest: Path) -> Path:
    raw = Path(str(record["features"]))
    return raw if raw.is_absolute() else (manifest.parent / raw)


def _relativize(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path, base).replace(os.sep, "/")
    except ValueError:  # different drive on Windows
        return str(path)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    # -- load every manifest into one pool ---------------------------------- #
    pool: List[Dict[str, object]] = []
    for manifest in args.manifests:
        records = read_jsonl(manifest)
        for record in records:
            record["_path"] = _feature_path(record, manifest)
            pool.append(record)
        print("loaded " + str(len(records)) + " from " + str(manifest))

    if not pool:
        raise SystemExit("No utterances loaded.")

    by_speaker: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in pool:
        by_speaker[str(record.get("speaker", "spk0"))].append(record)

    multilingual_speakers = {
        speaker: sorted({str(r["lang"]) for r in items})
        for speaker, items in by_speaker.items()
        if len({str(r["lang"]) for r in items}) > 1
    }

    out_base = args.output.parent
    out_base.mkdir(parents=True, exist_ok=True)

    pairs: List[Dict[str, object]] = []
    self_pairs = 0
    cross_pairs = 0
    singleton_speakers = 0

    for speaker, items in by_speaker.items():
        by_lang: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for record in items:
            by_lang[str(record["lang"])].append(record)

        if len(items) == 1:
            singleton_speakers += 1

        for target in items:
            target_lang = str(target["lang"])
            same_lang = [r for r in by_lang[target_lang] if r["id"] != target["id"]]
            other_lang = [
                r for r in items
                if str(r["lang"]) != target_lang
            ] if args.cross_lingual else []

            if args.max_prompt_code_len > 0:
                short_enough = [r for r in same_lang if int(r["code_len"]) <= args.max_prompt_code_len]
                if short_enough:
                    same_lang = short_enough
                short_enough = [r for r in other_lang if int(r["code_len"]) <= args.max_prompt_code_len]
                if short_enough:
                    other_lang = short_enough

            for _ in range(max(1, args.pairs_per_target)):
                use_cross = bool(other_lang) and rng.random() < args.cross_lingual_ratio
                candidates = other_lang if use_cross else same_lang
                if not candidates:
                    candidates = other_lang or same_lang

                if candidates:
                    prompt = rng.choice(candidates)
                elif args.allow_self_pairs:
                    prompt = target
                    self_pairs += 1
                else:
                    continue

                prompt_lang = str(prompt["lang"])
                is_cross = prompt_lang != target_lang
                cross_pairs += int(is_cross)

                pairs.append({
                    "id": str(target["id"]) + "__" + str(prompt["id"]),
                    "target_features": _relativize(target["_path"], out_base),
                    "prompt_features": _relativize(prompt["_path"], out_base),
                    "speaker": speaker,
                    "lang": target_lang,
                    "lang_alias": target.get("lang_alias"),
                    "prompt_lang": prompt_lang,
                    "cross_lingual": is_cross,
                    "text_len": int(target["text_len"]),
                    "code_len": int(target["code_len"]),
                })

    if not pairs:
        raise SystemExit(
            "No pairs produced. Every speaker has a single utterance - rerun with "
            "--allow-self-pairs, or fix the speaker labels in step 1."
        )

    rng.shuffle(pairs)
    count = write_jsonl(args.output, pairs)

    lang_counts: Dict[str, int] = defaultdict(int)
    for pair in pairs:
        lang_counts[str(pair["lang"])] += 1

    print("")
    print("wrote " + str(count) + " pairs -> " + str(args.output))
    print("  speakers               : " + str(len(by_speaker)))
    print("  single-utterance spks  : " + str(singleton_speakers))
    print("  self pairs             : " + str(self_pairs))
    print("  cross-lingual pairs    : " + str(cross_pairs)
          + (" ({:.1f}%)".format(100.0 * cross_pairs / count) if count else ""))
    print("  per language           : " + ", ".join(
        k + "=" + str(v) for k, v in sorted(lang_counts.items())))
    if multilingual_speakers:
        sample = list(multilingual_speakers.items())[:5]
        print("  multilingual speakers  : " + str(len(multilingual_speakers))
              + "  e.g. " + ", ".join(s + str(langs) for s, langs in sample))
    elif args.cross_lingual:
        print("  NOTE: --cross-lingual had no effect - no speaker appears in two languages. "
              "Rely on replay manifests (--train-manifest ...::en@0.2) to preserve "
              "the base model's languages instead.")
    if self_pairs:
        print("  WARNING: " + str(self_pairs) + " pairs use the target as its own prompt. "
              "These teach the model to copy the conditioning; keep them rare.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
