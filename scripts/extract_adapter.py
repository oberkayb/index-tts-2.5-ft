#!/usr/bin/env python3
"""
Extract a language adapter (``.safetensors``) that runs on top of stock IndexTTS-2.5.

From a trainer checkpoint (exact LoRA factors):

    python scripts/extract_adapter.py --checkpoint runs/tr_lora/checkpoints/best.pt --out adapters/tr.safetensors

From a merged/exported gpt.pth (difference to stock + truncated SVD):

    python scripts/extract_adapter.py --merged checkpoints/tr/gpt.pth --out adapters/tr_v2.safetensors

The stock weights default to ``<model-dir>/gpt.pth``. ``text_head`` is never stored:
inference does not use it.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="trainer checkpoint (best.pt / step*.pt)")
    src.add_argument("--merged", type=Path, help="merged gpt.pth exported from a LoRA run")
    p.add_argument("--base", type=Path, default=None, help="stock gpt.pth (default: <model-dir>/gpt.pth)")
    p.add_argument("--repo", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lang", default="tr")
    p.add_argument("--rank", type=int, default=None, help="--merged only: SVD rank (default: detected)")
    p.add_argument("--normalizer", default="turkish", help="text frontend used in training (stored as metadata)")
    p.add_argument("--case", default="tr_lower")
    p.add_argument("--fp32", action="store_true", help="store fp32 instead of fp16")
    args = p.parse_args()

    import torch

    from itts25ft import env
    from itts25ft.adapter import adapter_size_mb, extract_from_merged, extract_from_trainer, save_adapter

    base = args.base
    if base is None:
        repo = env.bootstrap(args.repo)
        base = env.find_model_dir(args.model_dir, repo) / "gpt.pth"
    started = time.time()
    if args.checkpoint:
        lora, rows, replace, report = extract_from_trainer(base, args.checkpoint)
        source = {"type": "checkpoint", "path": str(args.checkpoint)}
    else:
        lora, rows, replace, report = extract_from_merged(base, args.merged, rank=args.rank)
        source = {"type": "merged", "path": str(args.merged), "sha256": sha256(args.merged)}

    if report["unexpected"]:
        print("!! changed tensors outside LoRA/rows/heads (not stored):", report["unexpected"][:10])
    if len(report["ranks"]) > 1 or (report["ranks"] and report["ranks"][0] >= 256):
        print("!! no clean low-rank structure; was this run trained with mode=lora? ranks:", report["ranks"])

    meta = {
        "lang": args.lang,
        "lang_token": "<|{}|>".format(args.lang),
        "normalizer": args.normalizer,
        "case": args.case,
        "rank": report["ranks"],
        "source": source,
        "base_model": "IndexTeam/IndexTTS-2.5",
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }
    out = save_adapter(args.out, lora, rows, replace, meta, dtype=torch.float32 if args.fp32 else torch.float16)

    print(">> adapter    :", out, "({:.1f} MB)".format(adapter_size_mb(out)))
    print("   LoRA       : {} modules, rank {}".format(len(lora), report["ranks"]))
    if args.merged:
        print("   SVD error  : worst relative {:.2e}".format(report["worst_rel_error"]))
    for name, (index, _) in rows.items():
        print("   rows       : {} -> {} rows".format(name, index.numel()))
    print("   replaced   :", ", ".join(replace) or "-")
    print("   took       : {:.0f} s".format(time.time() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
