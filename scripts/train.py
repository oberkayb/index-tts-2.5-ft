#!/usr/bin/env python3
"""
Step 4: finetune the IndexTTS-2.5 AR model on a new language.

What this trainer is careful about, and why:

  * **The language embedding is the primary parameter.**  It gets its own,
    hotter learning rate, and a gradient mask so that only the row you are
    training can move.  The five languages the release already speaks are
    mathematically untouched.
  * **The body is adapted, not rewritten.**  Default mode is LoRA on the GPT
    block projections; the base weights stay frozen and the adapters merge back
    on export.  ``partial`` and ``full`` are available when you have the data to
    justify them.
  * **The emotion branch is frozen by default.**  Timbre/emotion
    disentanglement is a property of those encoders; retraining them on a small
    monolingual corpus is how you lose it.
  * **Replay is first class.**  Pass extra manifests with a weight
    (``--train-manifest data/en_pairs.jsonl::en@0.2``) and validation reports
    per-language losses so you can watch for drift instead of discovering it
    after synthesis.

Example
-------
    python scripts/train.py \
        --train-manifest data/pairs_train.jsonl::tr \
        --train-manifest data/en_pairs.jsonl::en@0.15 \
        --val-manifest data/pairs_val.jsonl::tr \
        --lang tr --output-dir runs/tr_lora \
        --batch-size 8 --grad-accumulation 4 --learning-rate 1e-4 --amp bf16
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from itts25ft import env  # noqa: E402
from itts25ft.utils import (  # noqa: E402
    CheckpointManager, MetricAverager, human_time, pick_device, resolve_amp_dtype, set_seed,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finetune IndexTTS-2.5 for a new language.")

    p.add_argument("--config", type=Path, default=None,
                   help="YAML file whose keys mirror these flags; CLI values win.")
    p.add_argument("--repo", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--base-checkpoint", default=None,
                   help="Override the GPT weights to start from (default: <model-dir>/gpt.pth).")

    p.add_argument("--train-manifest", dest="train_manifests", action="append", default=[],
                   help="path[::lang[:alias]][@weight] - repeat for replay mixing.")
    p.add_argument("--val-manifest", dest="val_manifests", action="append", default=[])
    p.add_argument("--lang", required=False, help="Target language code (for reporting/guards).")
    p.add_argument("--lang-alias", default=None)
    p.add_argument("--lang-init-from", default=None,
                   help="Seed the new language row from this trained language (e.g. es).")
    p.add_argument("--lang-init-noise", type=float, default=0.0)
    p.add_argument("--train-all-lang-rows", action="store_true",
                   help="Disable the gradient mask and let every language row move.")

    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accumulation", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lang-lr-multiplier", type=float, default=10.0)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", default="bf16", help="bf16 | fp16 | none")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--trainable-mode", default="lora", choices=("lora", "partial", "full"))
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=float, default=64.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--lora-last-n-layers", type=int, default=0)
    p.add_argument("--train-last-n-layers", type=int, default=6)
    p.add_argument("--train-text-embedding", action="store_true")
    p.add_argument("--no-train-heads", dest="train_heads", action="store_false", default=True)
    p.add_argument("--train-emotion", dest="preserve_emotion", action="store_false", default=True,
                   help="Unfreeze the emotion encoder (not recommended).")

    p.add_argument("--text-loss-weight", type=float, default=0.2)
    p.add_argument("--mel-loss-weight", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--spk-noise-std", type=float, default=0.01,
                   help="Jitter on the CAMPPlus vector; keeps voice cloning from overfitting.")
    p.add_argument("--emo-dropout", type=float, default=0.1)
    p.add_argument("--emo-source", default="target", choices=("target", "prompt"))
    p.add_argument("--prompt-emo-prob", type=float, default=0.25)

    p.add_argument("--max-text-tokens", type=int, default=0, help="0 = take the model's limit.")
    p.add_argument("--max-code-tokens", type=int, default=0)
    p.add_argument("--bucket-multiplier", type=int, default=32)

    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--val-interval", type=int, default=1000)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--keep-checkpoints", type=int, default=3)
    p.add_argument("--export-every-save", action="store_true",
                   help="Also write an inference-ready gpt.pth at every save.")
    p.add_argument("--resume", default="", help="Checkpoint path, or 'auto'.")
    p.add_argument("--forgetting-guard", type=float, default=0.0,
                   help="Warn when a replay language's val loss rises this much above baseline.")

    args = p.parse_args(argv)

    if args.config is not None:
        from omegaconf import OmegaConf

        file_cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True) or {}
        explicit = {
            a.lstrip("-").split("=", 1)[0].replace("-", "_")
            for a in (argv or sys.argv[1:])
            if a.startswith("--")
        }
        for key, value in file_cfg.items():
            key = key.replace("-", "_")
            if key in explicit or not hasattr(args, key):
                continue
            if key in ("train_manifests", "val_manifests") and isinstance(value, str):
                value = [value]
            if key in ("output_dir",) and value is not None:
                value = Path(value)
            setattr(args, key, value)

    if not args.train_manifests:
        p.error("at least one --train-manifest is required")
    return args


def build_scheduler(optimizer, warmup_steps: int, total_steps: int, min_ratio: float):
    """Linear warmup into cosine decay, floored at ``min_ratio`` of the peak LR."""
    total_steps = max(total_steps, warmup_steps + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def trainable_state_dict(model) -> Dict[str, torch.Tensor]:
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}


@torch.no_grad()
def evaluate(model, loader, device, loss_cfg, amp_dtype) -> Dict[str, float]:
    from itts25ft.losses import compute_losses, total_loss

    model.eval()
    overall = MetricAverager()
    per_lang: Dict[str, MetricAverager] = {}

    for batch in loader:
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            text_loss, mel_loss, metrics = compute_losses(
                model, batch, device, loss_cfg, training=False
            )
        loss = total_loss(text_loss, mel_loss, loss_cfg)
        size = batch["text_ids"].size(0)
        values = {
            "loss": loss.item(),
            "text_loss": text_loss.item(),
            "mel_loss": mel_loss.item(),
            "mel_top1": metrics["mel_top1"],
        }
        overall.update(values, size)
        for code in set(batch["langs"]):
            count = sum(1 for c in batch["langs"] if c == code)
            per_lang.setdefault(code, MetricAverager()).update(values, count)

    model.train()
    out = overall.compute()
    for code, meter in per_lang.items():
        for key, value in meter.compute().items():
            out["lang/" + code + "/" + key] = value
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    repo = env.bootstrap(args.repo)
    model_dir = env.find_model_dir(args.model_dir, repo)
    cfg = env.load_model_config(model_dir)

    from torch.utils.tensorboard import SummaryWriter

    from itts25ft.data import DatasetConfig, LengthBucketBatchSampler, ManifestSpec, PairedDataset, collate
    from itts25ft.lang import PRETRAINED_LANGS, language_dict, resolve
    from itts25ft.losses import LossConfig, compute_losses, total_loss
    from itts25ft.modeling import (
        LanguageEmbeddingGradMask, TrainableSpec, apply_trainable_spec, build_gpt,
        describe_trainable, export_inference_checkpoint, parameter_groups,
    )

    device = pick_device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(args.amp)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    ds_cfg = DatasetConfig(
        emo_source=args.emo_source,
        prompt_emo_prob=args.prompt_emo_prob,
        max_text_tokens=args.max_text_tokens or int(cfg.gpt.max_text_tokens),
        max_code_tokens=args.max_code_tokens or int(cfg.gpt.max_mel_tokens),
    )
    train_specs = [ManifestSpec.parse(spec) for spec in args.train_manifests]
    train_ds = PairedDataset(train_specs, ds_cfg)
    print(">> train pairs: " + str(len(train_ds)) + "  " + str(train_ds.language_counts()))

    val_ds = None
    if args.val_manifests:
        val_specs = [ManifestSpec.parse(spec) for spec in args.val_manifests]
        val_ds = PairedDataset(val_specs, ds_cfg)
        print(">> val pairs  : " + str(len(val_ds)) + "  " + str(val_ds.language_counts()))

    batch_sampler = LengthBucketBatchSampler(
        train_ds, args.batch_size, bucket_multiplier=args.bucket_multiplier, seed=args.seed
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=batch_sampler, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
            num_workers=max(0, args.num_workers // 2), pin_memory=device.type == "cuda",
        )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    print(">> loading base GPT ...")
    model = build_gpt(cfg, model_dir, device=device, checkpoint=args.base_checkpoint)

    target_slot = resolve(args.lang, args.lang_alias) if args.lang else None
    if target_slot is not None:
        from itts25ft.modeling import init_language_row

        source_id = None
        if args.lang_init_from:
            source_id = language_dict()[args.lang_init_from.lower()]
        note = init_language_row(model, target_slot.lang_id, source_id, args.lang_init_noise)
        print(">> language row: " + target_slot.describe() + " - " + note)

    spec = TrainableSpec(
        mode=args.trainable_mode,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_last_n_layers=args.lora_last_n_layers,
        train_last_n_layers=args.train_last_n_layers,
        train_text_embedding=args.train_text_embedding,
        train_heads=args.train_heads,
        preserve_emotion=args.preserve_emotion,
    )
    info = apply_trainable_spec(model, spec)
    model.to(device)
    print(">> trainable mode: " + str(info["mode"])
          + ("  (" + str(info["lora_modules"]) + " LoRA modules)" if info["lora_modules"] else ""))
    print(describe_trainable(model))

    grad_mask = None
    if target_slot is not None and not args.train_all_lang_rows:
        trainable_rows = [target_slot.lang_id]
        grad_mask = LanguageEmbeddingGradMask(model, trainable_rows)
        print(">> lang_embedding gradient restricted to row(s): " + str(trainable_rows))

    # ------------------------------------------------------------------ #
    # Optimisation
    # ------------------------------------------------------------------ #
    groups = parameter_groups(
        model, args.learning_rate, args.lang_lr_multiplier, args.weight_decay
    )
    if not groups:
        raise SystemExit("Nothing is trainable - check --trainable-mode and the freeze patterns.")
    optimizer = AdamW(groups, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8)

    steps_per_epoch = max(1, len(batch_sampler) // max(1, args.grad_accumulation))
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    scheduler = build_scheduler(optimizer, args.warmup_steps, total_steps, args.min_lr_ratio)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    loss_cfg = LossConfig(
        text_weight=args.text_loss_weight,
        mel_weight=args.mel_loss_weight,
        label_smoothing=args.label_smoothing,
        spk_noise_std=args.spk_noise_std,
        emo_dropout=args.emo_dropout,
    )

    ckpt_manager = CheckpointManager(output_dir / "checkpoints", keep=args.keep_checkpoints)
    writer = SummaryWriter(log_dir=str(output_dir / "logs"))
    (output_dir / "train_args.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ #
    # Resume
    # ------------------------------------------------------------------ #
    global_step = 0
    start_epoch = 0
    baselines: Dict[str, float] = {}
    resume_path = None
    if args.resume == "auto":
        resume_path = ckpt_manager.latest()
    elif args.resume:
        resume_path = Path(args.resume)
    if resume_path and Path(resume_path).is_file():
        state = torch.load(resume_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state["trainable"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        global_step = int(state.get("step", 0))
        start_epoch = int(state.get("epoch", 0))
        baselines = dict(state.get("baselines", {}))
        print(">> resumed from " + str(resume_path) + " at step " + str(global_step)
              + " (" + str(len(unexpected)) + " unexpected keys)")

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    model.train()
    meter = MetricAverager()
    started = time.time()
    last_log = time.time()
    stop = False

    print("")
    print(">> " + str(total_steps) + " optimiser steps ("
          + str(steps_per_epoch) + "/epoch), batch " + str(args.batch_size)
          + " x " + str(args.grad_accumulation) + " accumulation, amp=" + args.amp)
    print("")

    for epoch in range(start_epoch, args.epochs):
        batch_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)

        for micro_step, batch in enumerate(train_loader):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                text_loss, mel_loss, metrics = compute_losses(
                    model, batch, device, loss_cfg, training=True
                )
                loss = total_loss(text_loss, mel_loss, loss_cfg)

            scaler.scale(loss / args.grad_accumulation).backward()

            meter.update({
                "loss": loss.item(),
                "text_loss": text_loss.item(),
                "mel_loss": mel_loss.item(),
                "mel_top1": metrics["mel_top1"],
            }, batch["text_ids"].size(0))

            if (micro_step + 1) % args.grad_accumulation != 0:
                continue

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            # -- logging -------------------------------------------------- #
            if global_step % args.log_interval == 0:
                values = meter.compute()
                meter.reset()
                elapsed = time.time() - last_log
                last_log = time.time()
                rate = args.log_interval / max(elapsed, 1e-6)
                remaining = (total_steps - global_step) / max(rate, 1e-6)
                lr = scheduler.get_last_lr()[0]
                print(
                    "step {:>7} | loss {:.4f} | mel {:.4f} | text {:.4f} | top1 {:.3f} "
                    "| lr {:.2e} | {:.2f} it/s | eta {}".format(
                        global_step, values.get("loss", 0.0), values.get("mel_loss", 0.0),
                        values.get("text_loss", 0.0), values.get("mel_top1", 0.0),
                        lr, rate, human_time(remaining),
                    )
                )
                for key, value in values.items():
                    writer.add_scalar("train/" + key, value, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                for group in optimizer.param_groups:
                    if group.get("name") == "lang_embedding":
                        writer.add_scalar("train/lr_lang", group["lr"], global_step)

            # -- validation ----------------------------------------------- #
            if val_loader is not None and args.val_interval and global_step % args.val_interval == 0:
                val_metrics = evaluate(model, val_loader, device, loss_cfg, amp_dtype)
                line = "  val @ {:>7} | ".format(global_step) + " | ".join(
                    "{} {:.4f}".format(k, v) for k, v in sorted(val_metrics.items())
                    if "/" not in k
                )
                print(line)
                for key, value in val_metrics.items():
                    writer.add_scalar("val/" + key, value, global_step)

                for key, value in val_metrics.items():
                    if not key.startswith("lang/") or not key.endswith("/loss"):
                        continue
                    code = key.split("/")[1]
                    if code not in baselines:
                        baselines[code] = value
                        print("    baseline {:<4} loss {:.4f}".format(code, value))
                    else:
                        delta = value - baselines[code]
                        marker = ""
                        if (args.forgetting_guard > 0 and delta > args.forgetting_guard
                                and (target_slot is None or code != target_slot.code)):
                            marker = "  <-- FORGETTING: " + code + " drifted +{:.3f}".format(delta)
                        print("    {:<4} loss {:.4f} ({:+.4f}){}".format(code, value, delta, marker))
                        writer.add_scalar("val_drift/" + code, delta, global_step)

            # -- checkpointing -------------------------------------------- #
            if args.save_interval and global_step % args.save_interval == 0:
                path = ckpt_manager.save({
                    "trainable": trainable_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "step": global_step,
                    "epoch": epoch,
                    "baselines": baselines,
                    "spec": vars(spec),
                }, global_step)
                print("  saved " + str(path))
                if args.export_every_save:
                    import copy

                    snapshot = copy.deepcopy(model).float()
                    result = export_inference_checkpoint(
                        snapshot, output_dir / "exported" / ("gpt_step" + str(global_step) + ".pth"),
                        metadata={"step": global_step, "lang": args.lang},
                    )
                    del snapshot
                    print("  exported " + str(result["path"]))

            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break

        if stop:
            break

    # ------------------------------------------------------------------ #
    # Finish
    # ------------------------------------------------------------------ #
    if val_loader is not None:
        final_metrics = evaluate(model, val_loader, device, loss_cfg, amp_dtype)
        print("")
        print("final validation:")
        for key, value in sorted(final_metrics.items()):
            print("  {:<28}: {:.4f}".format(key, value))

    ckpt_manager.save({
        "trainable": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "step": global_step,
        "epoch": args.epochs,
        "baselines": baselines,
        "spec": vars(spec),
    }, global_step)

    result = export_inference_checkpoint(
        model, output_dir / "exported" / "gpt.pth",
        metadata={
            "step": global_step,
            "lang": args.lang,
            "lang_id": target_slot.lang_id if target_slot else None,
            "vocabulary_slot": target_slot.slot if target_slot else None,
            "mode": args.trainable_mode,
        },
    )
    writer.close()

    print("")
    print("training finished in " + human_time(time.time() - started))
    print("inference weights: " + str(result["path"])
          + "  (merged " + str(result["merged_adapters"]) + " adapters)")
    print("")
    print("Copy it over your model dir's gpt.pth (keep a backup!) and synthesise with:")
    print("  python scripts/synthesize.py --gpt-checkpoint " + str(result["path"])
          + " --lang " + str(args.lang) + " --prompt-audio ref.wav --text \"...\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
