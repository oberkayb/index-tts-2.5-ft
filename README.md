# itts25ft — Fine-tune a new language on IndexTTS-2.5

[English](#english) · [Türkçe](#türkçe)

> **Acknowledgements.** This project was inspired by [JarodMica/index-tts `training_v2`](https://github.com/JarodMica/index-tts/tree/training_v2) — especially the manifest → preprocess → pairs → train workflow and the idea of keeping fine-tuning code outside upstream. It was rebuilt from scratch for **IndexTTS-2.5** (CAMPPlus conditioning, language embeddings, EnhancedCodec, tiktoken vocab).

---

## English

Standalone fine-tuning for adding a **new language** to [IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) without breaking the five shipped languages (zh, en, ja, es, ar), **cross-lingual voice cloning**, or **timbre/emotion disentanglement**.

Does **not** patch the upstream repo (`index-tts-2.5/`); it only imports it. `git pull` on upstream stays conflict-free.

```
python scripts/check_setup.py --lang tr --load-model        # 0. verify
python scripts/prepare_manifest.py ...                      # 1. manifest
python scripts/preprocess.py ...                            # 2. feature cache
python scripts/build_pairs.py ...                           # 3. prompt/target pairs
python scripts/train.py --config configs/turkish.yaml       # 4. train
python scripts/synthesize.py --cross-lingual-check ...      # 5. test
```

### 1. Why a new trainer? (2 → 2.5 architecture)

The trainer in `index-tts-2-ft` targets IndexTTS-2 and **does not work** on 2.5. Verified differences:

| | IndexTTS-2 | IndexTTS-2.5 |
|---|---|---|
| Speaker conditioning | `conformer_perceiver` → **32 latents** (from mel) | **CAMPPlus 192-d** → `spk_emb_proj` → **1 latent** |
| Conditioning length | 32 + 2 = **34 positions** | 1 + 2 = **3 positions** |
| Language conditioning | none | `lang_embedding: Embedding(107, dim)` added at **every text position** |
| Text tokenizer | SentencePiece `bpe.model` (12k) | Whisper **tiktoken** `multilingual_zh_ja_yue_char_del` |
| Language control token | none | `<\|tr\|>` special token at text start |
| Semantic codec | MaskGCT codec | `EnhancedCodec` (`downsample_scale=2`) |
| `speed_emb` | present | **absent** in campplus mode (2 zero positions) |

> Note: `indextts/gpt/model_v2_5.py` is dead code — never imported. The real model is `UnifiedVoice` in `indextts/gpt/model_v2.py` with `spk_cond_mode="campplus"`, as used by `infer_v2_5.py`.

Practical consequences:

* The old trainer’s precomputed `[32, dim]` conditioning latent is meaningless. Cache the **raw 192-d CAMPPlus vector** instead.
* `UnifiedVoice.forward` in campplus mode does **not** apply `lang_embedding` (that path produces s2mel latents). Training forward is implemented in `losses.py` via `prepare_gpt_inputs` to match inference exactly.
* **No vocab surgery** for Turkish: `<|tr|>` already exists, `lang_to_token("tr") = 9`, and row 9 of `lang_embedding` is in the checkpoint — just untrained. That row is what we fine-tune.

### 2. Setup

```bash
# Install into the index-tts-2.5 environment (torch, transformers, tiktoken, etc.).
# This project only adds tensorboard/tqdm.
pip install -r requirements.txt

# Repo path is auto-detected (sibling directory). Otherwise:
export INDEXTTS25_REPO=/path/to/index-tts-2.5
export INDEXTTS25_MODEL_DIR=/path/to/index-tts-2.5/checkpoints
```

Download weights if needed:

```bash
cd ../index-tts-2.5
hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints
```

Verify **before** spending GPU time:

```bash
python scripts/check_setup.py --lang tr --normalizer turkish --case tr_lower \
    --sample-text data/tr_ornek_metinler.txt --load-model
```

Reports repo/checkpoint version, language slot, tokenizer efficiency (tokens/char), token id overflow, and whether the target `lang_embedding` row is still empty.

### 3. Data requirements

| | Minimum | Comfortable | Good |
|---|---|---|---|
| Duration | 5 h | 20–50 h | 100 h+ |
| Speakers | 10 | 50+ | 200+ |
| Per clip | 1–20 s, clean, single speaker | | |

**Speaker labels are critical.** Prompt (reference audio) and target (text to speak) must be **different clips from the same speaker**. Using the same clip for both teaches copying; prompt text leaks into output. `build_pairs.py` builds correct pairs and warns on self-pairs.

Audio: 16 kHz+ (24 kHz ideal), mono, normalized, trimmed silence. Transcripts must match audio word-for-word.

### 4. Step by step

#### 4.1 Manifest

```bash
python scripts/prepare_manifest.py \
    --format csv --input data/tr/metadata.csv --columns audio,text,speaker \
    --audio-dir data/tr/wavs --lang tr \
    --output data/tr/utterances.jsonl
```

`--format`: `jsonl`, `csv`, `tsv`, `ljspeech`, `folders`. No speaker column? Use `--speaker-from-path 1`.

#### 4.2 Feature extraction (GPU)

```bash
python scripts/preprocess.py \
    --manifest data/tr/utterances.jsonl \
    --output-dir data/tr/processed \
    --lang tr --normalizer turkish --case tr_lower \
    --min-seconds 1.0 --max-seconds 20.0 --val-size 256
```

Each utterance → one `.npz`: `codes`, `text_ids`, `spk_emb`, `emo_vec`. Training never opens wav files again. Re-runs skip existing files (`--overwrite` to force).

Extraction matches `infer_v2_5`: w2v-BERT layer 17 + shipped mean/var, `EnhancedCodec.quantize`, 80-bin Kaldi fbank → CAMPPlus, frozen emotion encoder.

#### 4.3 Prompt/target pairs

```bash
python scripts/build_pairs.py \
    --manifest data/tr/processed/utterances_train.jsonl \
    --output data/tr/pairs_train.jsonl --pairs-per-target 2

python scripts/build_pairs.py \
    --manifest data/tr/processed/utterances_val.jsonl \
    --output data/tr/pairs_val.jsonl --pairs-per-target 1
```

For speakers with recordings in multiple languages, add `--cross-lingual` (prompt lang A, target lang B) — the only **direct** supervision for “clone this voice, speak that language”.

#### 4.4 Training

```bash
python scripts/train.py --config configs/turkish.yaml
```

Or explicit flags:

```bash
python scripts/train.py \
    --train-manifest data/tr/pairs_train.jsonl::tr \
    --train-manifest data/replay/en_pairs.jsonl::en@0.15 \
    --val-manifest data/tr/pairs_val.jsonl::tr \
    --val-manifest data/replay/en_pairs_val.jsonl::en \
    --lang tr --lang-init-from es \
    --output-dir runs/tr_lora \
    --trainable-mode lora --lora-rank 32 \
    --batch-size 8 --grad-accumulation 4 \
    --learning-rate 1e-4 --amp bf16 \
    --forgetting-guard 0.05
```

Manifest syntax: `path[::lang[:alias]][@weight]`.

Export merged weights for inference:

```bash
python scripts/export.py \
    --checkpoint runs/tr_lora/checkpoints/step29000.pt \
    --output runs/tr_lora/exported/gpt.pth \
    --lang tr
```

#### 4.5 Test

```bash
python scripts/synthesize.py \
    --gpt-checkpoint runs/tr_lora/exported/gpt.pth \
    --lang tr --normalizer turkish --case tr_lower \
    --prompt-audio ref.wav \
    --text "Merhaba, bugün hava çok güzel." \
    --output out/tr.wav \
    --cross-lingual-check
```

`--cross-lingual-check` synthesizes short EN/ZH/JA/ES probes with the **same reference voice**. Compare to stock model — regression means fine-tune leaked into base languages.

### 5. Preserving cross-lingual behavior

The main risk when adding a language is **catastrophic forgetting**. Five layers of defense:

1. **Language row isolation.** `LanguageEmbeddingGradMask` keeps gradients only on the target row. Other 106 rows stay fixed (smoke-tested).
2. **LoRA (default).** GPT body frozen; rank-32 adapters on `c_attn`/`c_proj`/`c_fc` (~1% params). Adapters **merge** on export → plain `gpt.pth` for stock `infer_v2_5.py`.
3. **Replay.** Mix old-language data (`::en@0.15`). Per-language val loss; `--forgetting-guard 0.05` warns before you hear it in synthesis.
4. **Frozen emotion branch.** `emo_conditioning_encoder`, `emo_perceiver_encoder`, `emovec_layer`, `emo_layer`, `spk_emb_proj` stay frozen by default.
5. **Speaker vector augmentation.** `--spk-noise-std 0.01` reduces memorization on small corpora.

**Honest caveat:** without same-speaker cross-lingual recordings, there is no direct loss for “speak language B with voice from language A”. That capability **already exists in the base model** (CAMPPlus timbre + separate lang embedding). The job is **not to teach it, but not to break it**.

### 6. Hyperparameter guide

| Scenario | mode | LR | rank | epochs |
|---|---|---|---|---|
| 5–20 h, many speakers | `lora` | 1e-4 | 16–32 | 10–15 |
| 20–100 h | `lora` | 1e-4 | 32–64 | 8–12 |
| 100 h+, base langs don’t matter | `partial` | 2e-5 | — | 5–8 |
| Single speaker / voice clone | `lora` | 5e-5 | 8–16 | 5–10 |

* **`--lang-lr-multiplier 10`**: one new embedding row needs a higher LR than the LoRA body. Don’t lower the default.
* **`--lang-init-from`**: copy an existing row (e.g. `es` for Latin-script languages).
* **`--text-loss-weight 0.2`**: auxiliary text CE. Don’t raise to 1.0.
* **Effective batch**: aim for `batch_size × grad_accumulation ≥ 32`.
* **VRAM**: batch 8 + bf16 + LoRA ≈ 24 GB. Tight? `--batch-size 4 --grad-accumulation 8`.

### 7. Troubleshooting

| Issue | Fix |
|---|---|
| Could not locate IndexTTS-2.5 | Set `INDEXTTS25_REPO` or pass `--repo` |
| config.yaml version=2.0 | Download IndexTTS-2.5 weights |
| Token id overflow | Re-download checkpoints (wrong tiktoken vocab) |
| tokens/char > 0.5 | Improve text normalization (numbers, abbreviations) |
| Loss down, audio bad | Frontend mismatch — same `--normalizer`/`--case` in preprocess and synthesize |
| Prompt text in output | Self-pairs — fix speaker labels |
| Base languages degraded | Lower LR, more replay, smaller rank, `--lora-last-n-layers 8` |
| NaN / crash | Use `bf16` not `fp16`; `--grad-clip 1.0`; halve LR |

### 8. File map

```
itts25ft/
  env.py         repo/checkpoint discovery, 2.5 version check
  lang.py        language slot (<|xx|> token + embedding row), alias
  textfront.py   inference-parity text frontend + Turkish normalizer
  extractors.py  w2v-BERT / EnhancedCodec / CAMPPlus / emotion cache
  data.py        pair manifests, replay mixing, length bucketing
  modeling.py    model build, lang row init + grad mask, LoRA, export
  losses.py      2.5 training forward (lang_embedding) + masked CE
  utils.py       seed, checkpoint rotation, metric averaging
scripts/         pipeline steps 0–5 (above)
tests/smoke_test.py   30 checks without downloading weights
configs/turkish.yaml  reference training config (Turkish)
```

After code changes:

```bash
python tests/smoke_test.py
```

### Published checkpoint

A Turkish LoRA export (step 29000) may be published separately on Hugging Face — only `gpt.pth` is replaced; all other weights come from [IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5).

---

## Türkçe

IndexTTS-2.5'e **yeni bir dil eklemek**, bunu yaparken modelin hâlihazırda konuştuğu beş dili (zh, en, ja, es, ar), **cross-lingual ses klonlamayı** ve **tını/duygu ayrışmasını** bozmamak için yazılmış bağımsız bir eğitim projesi.

> **Teşekkür.** Bu proje [JarodMica/index-tts `training_v2`](https://github.com/JarodMica/index-tts/tree/training_v2) dalından esinlenilerek yapıldı — özellikle manifest → preprocess → pairs → train akışı ve fine-tuning kodunu upstream dışında tutma fikri. **IndexTTS-2.5** için (CAMPPlus koşullama, dil embedding’leri, EnhancedCodec, tiktoken vocab) sıfırdan yeniden yazıldı.

Upstream repoyu (`index-tts-2.5/`) hiç değiştirmez; onu sadece import eder. `git pull` yaptığınızda çakışma olmaz.

```
python scripts/check_setup.py --lang tr --load-model        # 0. doğrula
python scripts/prepare_manifest.py ...                      # 1. manifest
python scripts/preprocess.py ...                            # 2. feature cache
python scripts/build_pairs.py ...                           # 3. prompt/hedef çiftleri
python scripts/train.py --config configs/turkish.yaml       # 4. eğitim
python scripts/synthesize.py --cross-lingual-check ...      # 5. test
```

### 1. Neden yeni bir trainer? (2 → 2.5 mimari farkları)

`index-tts-2-ft` içindeki trainer IndexTTS-2 için yazılmış ve 2.5'te **çalışmaz**. Kaynak kodda doğrulanan farklar:

| | IndexTTS-2 | IndexTTS-2.5 |
|---|---|---|
| Konuşmacı koşullama | `conformer_perceiver` → **32 latent** (mel'den) | **CAMPPlus 192-d** embedding → **1 latent** |
| Koşullama uzunluğu | 32 + 2 = **34 pozisyon** | 1 + 2 = **3 pozisyon** |
| Dil koşullama | yok | `lang_embedding: Embedding(107, dim)`, her text pozisyonuna eklenir |
| Text tokenizer | SentencePiece `bpe.model` (12k) | Whisper **tiktoken** |
| Dil kontrol tokenı | yok | metnin başına `<\|tr\|>` |
| Semantic codec | MaskGCT codec | `EnhancedCodec` |

Pratik sonuç: eski trainer’ın `[32, dim]` latent’i anlamsız; **192-d CAMPPlus vektörü** cache’lenmeli. Türkçe için vocab ameliyatı **gerekmez** — `<|tr|>` ve `lang_embedding` satır 9 zaten checkpoint’te; eğitilmemiş hâlde duruyor.

### 2. Kurulum

```bash
pip install -r requirements.txt
export INDEXTTS25_REPO=/path/to/index-tts-2.5
export INDEXTTS25_MODEL_DIR=/path/to/index-tts-2.5/checkpoints

cd ../index-tts-2.5
hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints

python scripts/check_setup.py --lang tr --normalizer turkish --case tr_lower --load-model
```

### 3. Veri gereksinimleri

| | Minimum | Rahat | İyi |
|---|---|---|---|
| Süre | 5 saat | 20–50 saat | 100 saat+ |
| Konuşmacı | 10 | 50+ | 200+ |

Prompt ve hedef **aynı konuşmacının farklı kayıtları** olmalı. Self-pair prompt metninin çıktıya sızmasına yol açar.

### 4. Adım adım

Manifest → `preprocess.py` (`.npz` cache) → `build_pairs.py` → `train.py` → `export.py` → `synthesize.py`.

Manifest sözdizimi: `yol[::dil[:alias]][@ağırlık]`.

Export:

```bash
python scripts/export.py \
    --checkpoint runs/tr_lora/checkpoints/step29000.pt \
    --output runs/tr_lora/exported/gpt.pth \
    --lang tr
```

Test (frontend uyumu şart):

```bash
python scripts/synthesize.py \
    --gpt-checkpoint runs/tr_lora/exported/gpt.pth \
    --lang tr --normalizer turkish --case tr_lower \
    --prompt-audio ref.wav --text "Merhaba." \
    --output out/tr.wav --cross-lingual-check
```

### 5. Cross-lingual’i korumak

Beş savunma: (1) dil satırı gradyan maskesi, (2) LoRA + merge export, (3) replay, (4) donuk duygu dalı, (5) konuşmacı vektörü gürültüsü. İşimiz yeni yeteneği **öğretmek değil, bozmamak**.

### 6. Hiperparametre

Varsayılan: `lora` rank 32, LR 1e-4, `--lang-init-from es`, `--lang-lr-multiplier 10`, efektif batch ≥ 32. Detaylar İngilizce bölüm §6’da.

### 7. Sorun giderme

Loss düşüyor ama ses bozuksa → `preprocess.py` ile `synthesize.py`’de **aynı** `--normalizer` ve `--case`. Eski diller bozulduysa → LR↓, replay↑, rank↓.

### 8. Dosya haritası

```
itts25ft/          çekirdek kütüphane (env, lang, textfront, extractors, data, modeling, losses)
scripts/           pipeline 0–5
tests/smoke_test.py
configs/turkish.yaml
```

Değişiklik sonrası: `python tests/smoke_test.py`

Yayınlanan checkpoint yalnızca `gpt.pth` değiştirir; `feat1.pt`, `s2mel.pth`, `codec.pth` vb. [IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) tabanından gelir.
