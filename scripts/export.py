#!/usr/bin/env python3
"""
Turn a training checkpoint into a drop-in ``gpt.pth``.

Training checkpoints only store the trainable tensors (with LoRA that is a few
megabytes rather than several gigabytes), so exporting means: rebuild the base
model, re-apply the same trainable spec, load the delta, merge the adapters and
write a full state dict that stock ``infer_v2_5.py`` loads unmodified.

Example
-------
    python scripts/export.py \
        --checkpoint runs/tr_lora/checkpoints/step12000.pt \
        --output runs/tr_lora/exported/gpt_step12000.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from itts25ft import env  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export an inference-ready gpt.pth.")
    p.add_argument("--repo", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--base-checkpoint", default=None)
    p.add_argument("--checkpoint", type=Path, required=True, help="Training checkpoint (.pt).")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lang", default=None, help="Recorded in the export metadata.")
    p.add_argument("--lang-alias", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo = env.bootstrap(args.repo)
    model_dir = env.find_model_dir(args.model_dir, repo)
    cfg = env.load_model_config(model_dir)

    from itts25ft.modeling import (
        TrainableSpec, apply_trainable_spec, build_gpt, export_inference_checkpoint,
    )

    state = torch.load(args.checkpoint, map_location="cpu")
    if "trainable" not in state:
        raise SystemExit(
            str(args.checkpoint) + " does not look like a trainer checkpoint "
            "(no 'trainable' key). If it is already a gpt.pth, no export is needed."
        )
    spec_dict = dict(state.get("spec", {}))
    spec_dict.pop("extra_train", None)
    spec_dict.pop("extra_freeze", None)
    spec = TrainableSpec(**spec_dict) if spec_dict else TrainableSpec()

    print(">> rebuilding base model (" + str(cfg.gpt_checkpoint) + ")")
    model = build_gpt(cfg, model_dir, device="cpu", checkpoint=args.base_checkpoint)

    print(">> re-applying trainable spec: mode=" + spec.mode)
    apply_trainable_spec(model, spec)

    missing, unexpected = model.load_state_dict(state["trainable"], strict=False)
    loaded = len(state["trainable"])
    print(">> loaded " + str(loaded) + " finetuned tensors "
          + "(" + str(len(unexpected)) + " unexpected)")
    if unexpected:
        print("   unexpected: " + str(list(unexpected)[:5]))

    metadata = {
        "source_checkpoint": str(args.checkpoint),
        "step": int(state.get("step", 0)),
        "epoch": int(state.get("epoch", 0)),
        "mode": spec.mode,
        "lang": args.lang,
    }
    result = export_inference_checkpoint(model, args.output, metadata=metadata)

    print("")
    print("wrote " + result["path"])
    print("  tensors          : " + str(result["tensors"]))
    print("  merged adapters  : " + str(result["merged_adapters"]))
    print("  metadata         : " + json.dumps(metadata, ensure_ascii=False))
    print("")
    print("Use it with scripts/synthesize.py --gpt-checkpoint " + result["path"])
    print("or copy it over <model-dir>/gpt.pth to make it the default (back up first).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
