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
  4. exported state dicts are loadable by a stock ``UnifiedVoice``,
  5. text-embedding row training touches only the selected rows,
  6. the Turkish normaliser and the manifest filters behave on real corner cases.

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
    EmbeddingRowGradMask, LanguageEmbeddingGradMask, TrainableSpec, apply_trainable_spec,
    export_inference_checkpoint, has_lora, init_language_row, init_text_rows_from_subwords,
    merge_lora, parameter_groups, select_text_rows,
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


class FakeEncoding:
    """Just enough of tiktoken for init_text_rows_from_subwords."""

    TOKENS = {50: " kaç".encode(), 51: b"<|tr|>", 52: b"\xff"}
    SPLITS = {"kaç": [20, 21], "kach": [22, 23]}

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        return self.TOKENS[token_id]

    def encode(self, text: str, allowed_special=()) -> list:
        return self.SPLITS.get(text, [99])


def test_text_rows() -> None:
    print("\n[7] text-embedding row training")
    model = build_tiny()
    weight = model.text_embedding.weight
    untrained = [21, 50, 51, 52]
    with torch.no_grad():
        weight.mul_(2.0 / weight.norm(dim=1, keepdim=True))       # "trained": norm 2
        weight[untrained] *= 0.05                                  # "untrained": norm 0.1

    counts = {50: 10, 51: 5, 20: 7, 52: 1}
    rows, stats = select_text_rows(model, counts, min_count=2)
    check("selects used untrained rows only", rows == [50, 51], str(rows) + " " + str(stats))

    init = init_text_rows_from_subwords(model, [50, 51, 52], FakeEncoding())
    expected = (weight[22] + weight[23]) / 2
    check("whole word seeded from respelled trained pieces",
          bool(torch.allclose(weight[50], expected, atol=1e-6)), str(init))
    check("control token and byte rows kept",
          init["kept_special"] == 1 and init["kept_no_pieces"] == 1)

    try:
        apply_trainable_spec(build_tiny(), TrainableSpec(train_text_embedding=True, train_text_rows=True))
        check("whole-table and row modes are exclusive", False)
    except ValueError:
        check("whole-table and row modes are exclusive", True)

    apply_trainable_spec(model, TrainableSpec(mode="lora", lora_rank=4, train_text_rows=True))
    check("text embedding is trainable", model.text_embedding.weight.requires_grad)
    mask = EmbeddingRowGradMask(model.text_embedding.weight, rows, name="text_embedding")

    groups = parameter_groups(model, base_lr=1e-4, lang_lr_multiplier=10.0, text_lr_multiplier=10.0)
    text_group = [g for g in groups if g["name"] == "text_embedding"]
    check("text rows get their own undecayed group",
          len(text_group) == 1 and text_group[0]["weight_decay"] == 0.0
          and abs(text_group[0]["lr"] - 1e-3) < 1e-12)
    optimizer = torch.optim.AdamW(groups, lr=1e-4)

    batch = fake_batch()
    batch["text_ids"][:, :3] = torch.tensor([50, 51, 20])
    before = model.text_embedding.weight.detach().clone()
    text_loss, mel_loss, _ = compute_losses(model, batch, torch.device("cpu"), LossConfig())
    total_loss(text_loss, mel_loss, LossConfig()).backward()
    optimizer.step()

    after = model.text_embedding.weight.detach()
    moved = (after - before).abs().sum(dim=1) > 0
    check("selected rows moved", bool(moved[50] and moved[51]))
    check("every other row is bit-identical", int(moved.sum().item()) == 2,
          str(int(moved.sum().item())) + " rows changed (row 20 was in the batch)")
    mask.remove()


def test_normalizer_and_manifest(tmp: Path) -> None:
    print("\n[8] Turkish normaliser and manifest filters")
    import json
    import subprocess

    from itts25ft.textfront import normalize_turkish

    cases = {
        "list percent no longer crashes": ("Real %1,5, bilemedin %2.", "Real yüzde bir virgül beş, bilemedin yüzde iki."),
        "sentence-final number is cardinal": ("Yaşım 25.", "Yaşım yirmi beş."),
        "number before a sentence opener is cardinal": ("Bunu 9. Yani son.", "Bunu dokuz. Yani son."),
        "ordinal before a name": ("1. Dünya Savaşı, 5. Ordu", "birinci Dünya Savaşı, beşinci Ordu"),
        "ordinal before a lowercase word": ("6. his", "altıncı his"),
        "real word 'yak.' kept": ("Ateşi yak. Sonra gel.", "Ateşi yak. Sonra gel."),
        "'yak.' before a number expanded": ("yak. 5 km", "yaklaşık beş kilometre"),
        "leading zero read digit by digit": ("Tel. 0532", "telefon sıfır beş üç iki"),
    }
    for name, (raw, want) in cases.items():
        try:
            got = normalize_turkish(raw)
        except Exception as exc:  # noqa: BLE001
            got = repr(exc)
        check(name, got == want, repr(got))

    audio = tmp / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    rows = [
        ("a", "Merhaba dünya, bugün hava çok güzel.", 2.5),     # kept
        ("b", "Merhaba dünya, bugün hava çok güzel.", 2.5),     # same text -> dropped
        ("c", "İzlediğiniz için teşekkürler.", 28.0),           # 1 char/s -> dropped
        ("d", "Uzun bir kayıt bu.", 40.0),                       # too long -> dropped
        ("e", "Altyazı M.K.", 2.0),                              # regex -> dropped
    ]
    source = tmp / "source.jsonl"
    with source.open("w", encoding="utf-8") as handle:
        for uid, text, dur in rows:
            (audio / (uid + ".mp3")).write_bytes(b"")
            handle.write(json.dumps({"audio": "audio/" + uid + ".mp3", "text": text,
                                     "speaker": "s1", "duration": dur}, ensure_ascii=False) + "\n")
    out = tmp / "manifests" / "utterances.jsonl"
    script = Path(__file__).resolve().parent.parent / "scripts" / "prepare_manifest.py"
    result = subprocess.run(
        [sys.executable, str(script), "--format", "jsonl", "--input", str(source),
         "--audio-dir", str(tmp), "--lang", "tr", "--output", str(out),
         "--max-duration", "30", "--min-cps", "5", "--max-same-text", "1",
         "--exclude-regex", r"altyaz[ıi]"],
        capture_output=True, text=True, encoding="utf-8",
    )
    kept = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()] if out.is_file() else []
    check("manifest filters keep exactly one clip", [r["id"] for r in kept] == ["a"],
          (result.stderr.strip().splitlines() or [""])[-1] if result.returncode else str([r["id"] for r in kept]))
    check("manifest carries duration", bool(kept) and kept[0].get("duration") == 2.5)
    check("relative audio path resolves from the manifest folder",
          bool(kept) and (out.parent / kept[0]["audio"]).resolve() == (audio / "a.mp3").resolve(),
          kept[0]["audio"] if kept else "")


def main() -> int:
    import tempfile

    print("=" * 66)
    print("itts25ft smoke test (tiny model, CPU)")
    print("=" * 66)

    test_forward_and_backward()
    test_conditioning_shape()
    test_language_isolation()
    test_lora_identity_and_merge()
    test_text_rows()
    with tempfile.TemporaryDirectory() as tmp:
        test_parameter_groups_and_export(Path(tmp))
        test_data_pipeline(Path(tmp))
        test_normalizer_and_manifest(Path(tmp))

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
