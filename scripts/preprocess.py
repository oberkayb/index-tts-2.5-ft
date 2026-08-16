#!/usr/bin/env python3
"""
Step 2: extract and cache every feature the AR model consumes.

For each utterance this writes one ``.npz`` holding ``codes``, ``text_ids``,
``spk_emb`` and ``emo_vec``, then emits train/val manifests carrying the lengths
the sampler needs.  Training after this never opens a wav file again.

The extraction path mirrors ``infer_v2_5`` exactly - w2v-BERT layer 17 with the
shipped mean/var statistics, ``EnhancedCodec.quantize``, 80-bin Kaldi fbank into
CAMPPlus, and the GPT's own frozen emotion encoder.  Re-running is cheap: files
that already exist are skipped unless ``--overwrite`` is given.

Example
-------
    python scripts/preprocess.py \
        --manifest data/tr/utterances.jsonl \
        --output-dir data/tr/processed \
        --lang tr --normalizer turkish --case tr_lower \
        --min-seconds 1.0 --max-seconds 20.0 --val-size 256
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from itts25ft import env  # noqa: E402
from itts25ft.utils import human_time, read_jsonl, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache IndexTTS-2.5 features for finetuning.")
    p.add_argument("--repo", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--manifest", type=Path, required=True, help="Utterance manifest from step 1.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lang", required=True)
    p.add_argument("--lang-alias", default=None)
    p.add_argument("--normalizer", default="none")
    p.add_argument("--case", default="lower")
    p.add_argument("--no-pronunciation", action="store_true",
                   help="Disable <word|PHONES> annotation handling.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-seconds", type=float, default=0.8)
    p.add_argument("--max-seconds", type=float, default=25.0)
    p.add_argument("--max-ref-seconds", type=float, default=15.0,
                   help="Clip used for speaker/emotion conditioning (upstream default 15 s).")
    p.add_argument("--val-size", type=int, default=128, help="Utterances held out for validation.")
    p.add_argument("--val-speakers", type=int, default=0,
                   help="Hold out whole speakers instead of random utterances.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Process at most N utterances (smoke tests).")
    p.add_argument("--report-every", type=int, default=200)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo = env.bootstrap(args.repo)
    model_dir = env.find_model_dir(args.model_dir, repo)
    cfg = env.load_model_config(model_dir)

    from itts25ft.lang import resolve
    from itts25ft.textfront import TextFrontend, TextFrontendConfig
    from itts25ft.extractors import FeatureExtractor, save_features
    from itts25ft.modeling import build_gpt, validate_text_vocab

    slot = resolve(args.lang, args.lang_alias)
    print(">> language: " + slot.describe())

    frontend = TextFrontend(
        model_dir,
        slot,
        TextFrontendConfig(
            normalizer=args.normalizer,
            case=args.case,
            apply_pronunciation=not args.no_pronunciation,
        ),
    )

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(">> loading GPT (frozen, used for emotion vectors) ...")
    gpt = build_gpt(cfg, model_dir, device=device)
    gpt.eval()
    for param in gpt.parameters():
        param.requires_grad = False

    print(">> loading w2v-BERT / codec / CAMPPlus ...")
    extractor = FeatureExtractor(model_dir, cfg, device=device, gpt=gpt)

    utterances = read_jsonl(args.manifest)
    if args.limit:
        utterances = utterances[: args.limit]
    print(">> " + str(len(utterances)) + " utterances to process")

    features_dir = args.output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    max_text = int(cfg.gpt.max_text_tokens)
    max_codes = int(cfg.gpt.max_mel_tokens)

    processed: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    skipped_short = skipped_long = skipped_text = cached = 0
    checked_vocab = False
    started = time.time()

    for index, record in enumerate(utterances):
        uid = str(record["id"])
        audio_path = Path(record["audio"])
        if not audio_path.is_absolute():
            audio_path = (args.manifest.parent / audio_path).resolve()

        out_path = features_dir / (uid + ".npz")
        try:
            text_ids = frontend.encode(str(record["text"]))
            if not checked_vocab:
                validate_text_vocab(gpt, max(text_ids))
                checked_vocab = True
            if len(text_ids) > max_text:
                skipped_text += 1
                continue

            if out_path.is_file() and not args.overwrite:
                with np.load(out_path) as data:
                    code_len = int(data["codes"].shape[0])
                    text_len = int(data["text_ids"].shape[0])
                cached += 1
            else:
                feats = extractor.process(
                    audio_path,
                    max_ref_seconds=args.max_ref_seconds,
                    max_target_seconds=args.max_seconds,
                )
                if feats.duration < args.min_seconds:
                    skipped_short += 1
                    continue
                code_len = int(feats.codes.shape[0])
                if code_len > max_codes:
                    skipped_long += 1
                    continue
                save_features(out_path, feats, np.asarray(text_ids, dtype=np.int32))
                text_len = len(text_ids)

            processed.append({
                "id": uid,
                "features": str(out_path.relative_to(args.output_dir)),
                "speaker": str(record.get("speaker", "spk0")),
                "lang": slot.code,
                "lang_alias": slot.slot if slot.aliased else None,
                "text": str(record["text"]),
                "text_len": text_len,
                "code_len": code_len,
            })
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
            failures.append({"id": uid, "audio": str(audio_path), "error": repr(exc)})
            if len(failures) <= 3:
                traceback.print_exc()
            continue

        if args.report_every and (index + 1) % args.report_every == 0:
            elapsed = time.time() - started
            rate = (index + 1) / max(elapsed, 1e-6)
            remaining = (len(utterances) - index - 1) / max(rate, 1e-6)
            print("  [{}/{}] {:.1f} utt/s, eta {}".format(
                index + 1, len(utterances), rate, human_time(remaining)))

    if not processed:
        raise SystemExit("Nothing was processed successfully; see the errors above.")

    # -- train / val split ------------------------------------------------- #
    rng = np.random.default_rng(args.seed)
    if args.val_speakers > 0:
        speakers = sorted({str(r["speaker"]) for r in processed})
        rng.shuffle(speakers)
        held_out = set(speakers[: args.val_speakers])
        val = [r for r in processed if str(r["speaker"]) in held_out]
        train = [r for r in processed if str(r["speaker"]) not in held_out]
    else:
        order = rng.permutation(len(processed))
        val_count = min(args.val_size, max(0, len(processed) // 10))
        val_idx = set(order[:val_count].tolist())
        val = [r for i, r in enumerate(processed) if i in val_idx]
        train = [r for i, r in enumerate(processed) if i not in val_idx]

    train_path = args.output_dir / "utterances_train.jsonl"
    val_path = args.output_dir / "utterances_val.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    stats = {
        "language": slot.code,
        "vocabulary_slot": slot.slot,
        "lang_id": slot.lang_id,
        "normalizer": args.normalizer,
        "case": args.case,
        "total_input": len(utterances),
        "processed": len(processed),
        "cached_reuse": cached,
        "train": len(train),
        "val": len(val),
        "skipped_short": skipped_short,
        "skipped_long": skipped_long,
        "skipped_text_too_long": skipped_text,
        "failures": len(failures),
        "speakers": len({str(r["speaker"]) for r in processed}),
        "code_len_mean": float(np.mean([r["code_len"] for r in processed])),
        "code_len_p95": float(np.percentile([r["code_len"] for r in processed], 95)),
        "text_len_mean": float(np.mean([r["text_len"] for r in processed])),
        "text_len_p95": float(np.percentile([r["text_len"] for r in processed], 95)),
    }
    (args.output_dir / "preprocess_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if failures:
        write_jsonl(args.output_dir / "preprocess_failures.jsonl", failures)

    print("")
    print("done in " + human_time(time.time() - started))
    for key, value in stats.items():
        print("  {:<22}: {}".format(key, value))
    print("")
    print("train manifest: " + str(train_path))
    print("val manifest  : " + str(val_path))
    print("next: python scripts/build_pairs.py --manifest " + str(train_path)
          + " --output " + str(args.output_dir / "pairs_train.jsonl"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
