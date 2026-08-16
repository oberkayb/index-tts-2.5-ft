#!/usr/bin/env python3
"""
Self-contained checks for the training math - no checkpoints, no GPU, seconds to run.

Builds a miniature ``UnifiedVoice`` in campplus mode and verifies the four things
that are easy to get subtly wrong and expensive to discover after a training run:

  1. the loss path runs and produces finite gradients,
  2. the language embedding actually participates in the graph, and the gradient
     mask confines updates to the rows we chose,
  3. LoRA adapters are exactly identity at init and merging them back into the
     base weights is numerically equivalent to running them,
  4. exported state dicts are loadable by a stock ``UnifiedVoice``.

    python tests/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from itts25ft import env  # noqa: E402

env.bootstrap()

from indextts.gpt.model_v2 import UnifiedVoice  # noqa: E402

from itts25ft.losses import LossConfig, compute_losses, total_loss  # noqa: E402
from itts25ft.modeling import (  # noqa: E402
    LanguageEmbeddingGradMask, TrainableSpec, apply_trainable_spec, export_inference_checkpoint,
    has_lora, init_language_row, merge_lora, parameter_groups,
)

MODEL_DIM = 64
HEADS = 4
NUM_TEXT_TOKENS = 128
NUM_MEL_CODES = 64
START_MEL, STOP_MEL = 62, 63

EMO_MODULE = {
    "output_size": 32,
    "linear_units": 64,
    "attention_heads": 4,
    "num_blocks": 1,
    "input_layer": "conv2d2",
    "perceiver_mult": 2,
}

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print("  PASS  " + name + ((" - " + detail) if detail else ""))
    else:
        FAILED.append(name)
        print("  FAIL  " + name + ((" - " + detail) if detail else ""))


def build_tiny() -> UnifiedVoice:
    torch.manual_seed(0)
    return UnifiedVoice(
        layers=2,
        model_dim=MODEL_DIM,
        heads=HEADS,
        max_text_tokens=48,
        max_mel_tokens=96,
        number_text_tokens=NUM_TEXT_TOKENS,
        number_mel_codes=NUM_MEL_CODES,
        start_mel_token=START_MEL,
        stop_mel_token=STOP_MEL,
        start_text_token=0,
        stop_text_token=1,
        mel_length_compression=1024,
        use_mel_codes_as_input=True,
        checkpointing=False,
        emo_condition_module=EMO_MODULE,
        spk_cond_mode="campplus",
    )


def fake_batch(batch_size: int = 3, text_len: int = 12, code_len: int = 20, lang_id: int = 9):
    torch.manual_seed(1)
    text_ids = torch.randint(2, NUM_TEXT_TOKENS - 1, (batch_size, text_len))
    codes = torch.randint(0, START_MEL - 1, (batch_size, code_len))
    text_lengths = torch.tensor([text_len, text_len - 3, text_len - 5][:batch_size])
    code_lengths = torch.tensor([code_len, code_len - 4, code_len - 7][:batch_size])
    return {
        "text_ids": text_ids,
        "codes": codes,
        "spk_emb": torch.randn(batch_size, 192),
        "emo_vec": torch.randn(batch_size, MODEL_DIM),
        "lang_ids": torch.full((batch_size,), lang_id, dtype=torch.long),
        "text_lengths": text_lengths,
        "code_lengths": code_lengths,
        "langs": ["tr"] * batch_size,
    }


def test_forward_and_backward() -> None:
    print("\n[1] loss path")
    model = build_tiny()
    model.train()
    batch = fake_batch()
    device = torch.device("cpu")

    text_loss, mel_loss, metrics = compute_losses(model, batch, device, LossConfig())
    check("losses are finite", bool(torch.isfinite(text_loss) and torch.isfinite(mel_loss)),
          "text={:.3f} mel={:.3f}".format(text_loss.item(), mel_loss.item()))
    check("mel_top1 in [0,1]", 0.0 <= metrics["mel_top1"] <= 1.0,
          "top1={:.3f}".format(metrics["mel_top1"]))

    loss = total_loss(text_loss, mel_loss, LossConfig())
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check("gradients produced", len(grads) > 0, str(len(grads)) + " tensors")
    check("gradients finite", all(torch.isfinite(g).all().item() for g in grads))
    check("lang_embedding got gradient", model.lang_embedding.weight.grad is not None
          and model.lang_embedding.weight.grad.abs().sum().item() > 0)


def test_conditioning_shape() -> None:
    print("\n[2] conditioning layout")
    from itts25ft.losses import build_conditioning

    model = build_tiny()
    conds = build_conditioning(model, torch.randn(3, 192), torch.randn(3, MODEL_DIM))
    check("3 conditioning positions", tuple(conds.shape) == (3, 3, MODEL_DIM), str(tuple(conds.shape)))
    check("last two positions are zero-padded",
          bool(torch.allclose(conds[:, 1:], torch.zeros_like(conds[:, 1:]))))


def test_language_isolation() -> None:
    print("\n[3] language row isolation")
    model = build_tiny()
    n_rows = model.lang_embedding.num_embeddings

    # Seeding a new row from a trained one must copy it exactly.
    init_language_row(model, target_lang_id=9, source_lang_id=3)
    check("row seeded from source",
          bool(torch.equal(model.lang_embedding.weight[9], model.lang_embedding.weight[3])))

    for param in model.parameters():
        param.requires_grad = True
    mask = LanguageEmbeddingGradMask(model, [9])

    batch = fake_batch(lang_id=9)
    text_loss, mel_loss, _ = compute_losses(model, batch, torch.device("cpu"), LossConfig())
    total_loss(text_loss, mel_loss, LossConfig()).backward()

    grad = model.lang_embedding.weight.grad
    other_rows = torch.cat([grad[:9], grad[10:]])
    check("target row has gradient", grad[9].abs().sum().item() > 0)
    check("every other language row is frozen", other_rows.abs().sum().item() == 0.0,
          str(n_rows - 1) + " rows masked")
    mask.remove()


def test_lora_identity_and_merge() -> None:
    print("\n[4] LoRA identity and merge")
    model = build_tiny()
    model.eval()
    batch = fake_batch()
    device = torch.device("cpu")

    with torch.no_grad():
        base_text, base_mel, _ = compute_losses(model, batch, device, LossConfig(), training=False)

    spec = TrainableSpec(mode="lora", lora_rank=4, lora_alpha=8.0)
    info = apply_trainable_spec(model, spec)
    check("adapters injected", info["lora_modules"] > 0, str(info["lora_modules"]) + " modules")
    check("only adapters + language + heads train",
          info["trainable"] < info["frozen"],
          "{:,} trainable vs {:,} frozen".format(info["trainable"], info["frozen"]))

    with torch.no_grad():
        lora_text, lora_mel, _ = compute_losses(model, batch, device, LossConfig(), training=False)
    check("zero-init adapters are identity",
          bool(torch.allclose(base_mel, lora_mel, atol=1e-6)),
          "delta={:.2e}".format(abs(base_mel.item() - lora_mel.item())))

    # Give the adapters a non-trivial value, then confirm merging is equivalent.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(".lora_B"):
                param.normal_(0, 0.02)

    with torch.no_grad():
        pre_text, pre_mel, _ = compute_losses(model, batch, device, LossConfig(), training=False)
    merged = merge_lora(model)
    with torch.no_grad():
        post_text, post_mel, _ = compute_losses(model, batch, device, LossConfig(), training=False)

    check("merge preserves output", bool(torch.allclose(pre_mel, post_mel, atol=1e-5)),
          "merged {} modules, delta={:.2e}".format(merged, abs(pre_mel.item() - post_mel.item())))
    check("no adapters remain", not has_lora(model))


def test_parameter_groups_and_export(tmp: Path) -> None:
    print("\n[5] optimizer groups and export")
    model = build_tiny()
    apply_trainable_spec(model, TrainableSpec(mode="lora", lora_rank=4))
    groups = parameter_groups(model, base_lr=1e-4, lang_lr_multiplier=10.0)
    names = {g["name"] for g in groups}
    check("language group exists", "lang_embedding" in names, str(sorted(names)))
    lang_group = [g for g in groups if g["name"] == "lang_embedding"][0]
    check("language LR is boosted", abs(lang_group["lr"] - 1e-3) < 1e-12,
          "lr={:.1e}".format(lang_group["lr"]))

    out = tmp / "gpt_export.pth"
    result = export_inference_checkpoint(model, out)
    check("export written", out.is_file(), str(result["tensors"]) + " tensors")

    state = torch.load(out, map_location="cpu")["model"]
    check("no adapter keys exported", not any(".lora_" in k for k in state))
    check("no inference_model keys exported", not any(k.startswith("inference_model.") for k in state))

    fresh = build_tiny()
    missing, unexpected = fresh.load_state_dict(state, strict=False)
    check("stock model loads the export", len(unexpected) == 0,
          str(len(missing)) + " missing / " + str(len(unexpected)) + " unexpected")


def test_data_pipeline(tmp: Path) -> None:
    print("\n[6] manifests, pairing and batching")
    import json

    import numpy as np

    from itts25ft.data import (
        DatasetConfig, LengthBucketBatchSampler, ManifestSpec, PairedDataset, collate,
    )

    spec = ManifestSpec.parse("data/tr.jsonl::az:tt@0.25")
    check("manifest spec parsing",
          spec.lang == "az" and spec.alias == "tt" and abs(spec.weight - 0.25) < 1e-9,
          "lang=" + str(spec.lang) + " alias=" + str(spec.alias) + " weight=" + str(spec.weight))

    features_dir = tmp / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(12):
        code_len = 20 + i
        text_len = 8 + (i % 5)
        path = features_dir / ("utt" + str(i) + ".npz")
        np.savez(
            path,
            codes=np.random.randint(0, START_MEL - 1, size=code_len).astype(np.int32),
            text_ids=np.random.randint(2, NUM_TEXT_TOKENS - 1, size=text_len).astype(np.int32),
            spk_emb=np.random.randn(192).astype(np.float32),
            emo_vec=np.random.randn(MODEL_DIM).astype(np.float32),
        )
        rows.append({
            "id": "pair" + str(i),
            "target_features": "features/utt" + str(i) + ".npz",
            "prompt_features": "features/utt" + str((i + 1) % 12) + ".npz",
            "speaker": "spk" + str(i % 3),
            "lang": "tr" if i % 3 else "en",
            "text_len": text_len,
            "code_len": code_len,
        })

    manifest = tmp / "pairs.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )

    dataset = PairedDataset(
        [ManifestSpec(path=manifest)], DatasetConfig(max_code_tokens=96, max_text_tokens=48),
        verbose=False,
    )
    check("all pairs loaded", len(dataset) == 12, str(len(dataset)) + " records")
    check("languages tracked", dataset.language_counts() == {"tr": 8, "en": 4},
          str(dataset.language_counts()))

    sampler = LengthBucketBatchSampler(dataset, batch_size=4, bucket_multiplier=2, seed=0)
    batches = list(iter(sampler))
    check("batches cover the dataset", sum(len(b) for b in batches) == 12,
          str(len(batches)) + " batches")

    batch = collate([dataset[i] for i in batches[0]])
    bsz = len(batches[0])
    check("collate pads text", batch["text_ids"].shape[0] == bsz,
          str(tuple(batch["text_ids"].shape)))
    check("lengths match padding",
          int(batch["text_lengths"].max()) == batch["text_ids"].shape[1]
          and int(batch["code_lengths"].max()) == batch["codes"].shape[1])
    check("lang ids resolved", batch["lang_ids"].dtype == torch.long
          and set(batch["lang_ids"].tolist()) <= {0, 9},
          str(sorted(set(batch["lang_ids"].tolist()))))
    check("spk_emb is raw 192-d", tuple(batch["spk_emb"].shape) == (bsz, 192),
          str(tuple(batch["spk_emb"].shape)))

    # The batch has to survive the real loss path unchanged.
    model = build_tiny()
    text_loss, mel_loss, _ = compute_losses(model, batch, torch.device("cpu"), LossConfig())
    check("real batch runs through the loss",
          bool(torch.isfinite(text_loss) and torch.isfinite(mel_loss)),
          "text={:.3f} mel={:.3f}".format(text_loss.item(), mel_loss.item()))


def main() -> int:
    import tempfile

    print("=" * 66)
    print("itts25ft smoke test (tiny model, CPU)")
    print("=" * 66)

    test_forward_and_backward()
    test_conditioning_shape()
    test_language_isolation()
    test_lora_identity_and_merge()
    with tempfile.TemporaryDirectory() as tmp:
        test_parameter_groups_and_export(Path(tmp))
        test_data_pipeline(Path(tmp))

    print("")
    print("=" * 66)
    print(str(len(PASSED)) + " passed, " + str(len(FAILED)) + " failed")
    if FAILED:
        for name in FAILED:
            print("  FAILED: " + name)
    print("=" * 66)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
