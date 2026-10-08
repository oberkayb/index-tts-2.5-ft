# itts25ft — Fine-tune a new language on IndexTTS-2.5

[English](#english) · [Türkçe](#türkçe)

> Inspired by [JarodMica/index-tts `training_v2`](https://github.com/JarodMica/index-tts/tree/training_v2), rebuilt from scratch for **IndexTTS-2.5** (CAMPPlus conditioning, language embeddings, EnhancedCodec, tiktoken vocab).

---

## English

Adds a **new language** (reference config: Turkish) to [IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) while keeping the shipped languages (zh, en, ja, es, ar), cross-lingual voice cloning and emotion control intact. The upstream repo is only imported, never patched.

### What's new in this version

- **Untrained text rows are trained (`--train-text-rows`).** The base model never trained the tiktoken rows Turkish uses (` kaç`, ` çok`, `ı`, `ş`, `<|tr|>`: norm ~0.1 vs 1–3), which made short words swap (` kaç` → ` evet`, ` tekrar` → ` belki`). Only those rows are trained: gradient-masked, own LR group, no weight decay, seeded from trained sub-word pieces. Enabled in `configs/turkish.yaml`.
- **Data quality:** `prepare_manifest.py` filters ASR corpora (`--max-duration`, `--min-cps/--max-cps`, `--max-same-text`, `--exclude-regex`). `preprocess.py` **drops** over-long clips instead of truncating audio with the full transcript (that taught word skipping).
- **Turkish normalizer:** numbers like `%1,5,` and `1.234,50`, ordinals vs. sentence ends (`25. Yani`), context-aware abbreviations (`yak. 5 km`, `Av. Mehmet`).
- **Training:** the text-row selection is saved to `text_rows.json` and stays identical on resume. The config is set to 4 epochs for the ~490 h corpus.
- **WebUI** (`webui_ft.py`): load trainer checkpoints without exporting, token preview showing trained/untrained rows.
- **MCP server** (`mcp_server.py`): use the model from Claude Desktop, Cursor, VS Code and other MCP clients.
- More smoke tests (`tests/smoke_test.py`).

Result on our Turkish run (490 h): WER on unseen speakers dropped from 7.1% (step 8000) to 6.3% (step 12000). The word swaps are gone, and the other languages pass the probes.

### Quick start

```bash
pip install -r requirements.txt                              # inside the index-tts-2.5 env
python scripts/check_setup.py --lang tr --load-model        # 0. verify repo, weights, language slot
python scripts/prepare_manifest.py ...                      # 1. manifest (+ quality filters)
python scripts/preprocess.py ...                            # 2. feature cache (.npz)
python scripts/build_pairs.py ...                           # 3. prompt/target pairs (same speaker, different clips)
python scripts/train.py --config configs/turkish.yaml       # 4. train (LoRA + language row + text rows)
python scripts/export.py --checkpoint <run>/checkpoints/best.pt --output gpt.pth --lang tr
python scripts/synthesize.py --gpt-checkpoint gpt.pth --lang tr --normalizer turkish --case tr_lower \
    --prompt-audio ref.wav --text "Merhaba." --output out.wav --cross-lingual-check
```

The repo is auto-detected as a sibling folder, or set `INDEXTTS25_REPO` / `INDEXTTS25_MODEL_DIR`. Use the same `--normalizer/--case` in preprocessing and synthesis.

### WebUI

```bash
webui.bat                        # http://127.0.0.1:7860 (GPU 0); --gpu 1 --port 7861 --qwen-emo
```

Pick any checkpoint (`best.pt`, `step*.pt`, exported `.pth`, stock). The token preview colours rows trained by the fine-tune green and rows that are still untrained red.

### MCP server

Tools: `synthesize`, `list_voices`, `list_checkpoints`, `load_checkpoint`. Put reference voices into `voices/` (file name = voice name). Output goes to `outputs/mcp/`.

```json
{ "mcpServers": { "indextts-tr": { "command": "uv", "args": [
  "run", "--quiet", "--project", "<path>/index-tts", "--with", "mcp==2.3.0",
  "python", "<path>/index-tts-2.5-ft/mcp_server.py", "--gpu", "0" ] } } }
```

To share one loaded model between several clients, run `mcp_server.bat --transport http --port 8765` and connect to `http://127.0.0.1:8765/mcp`.

### Notes

- **Data:** 20 h+ and 50+ speakers is comfortable. Prompt and target must be different clips of the same speaker.
- **Keeping old languages:** language-row gradient mask, LoRA merged on export, optional replay (`path::en@0.15`), frozen emotion branch.
- **Troubleshooting:** if short words get swapped, use `--train-text-rows`. If the loss looks fine but the audio is bad, the frontend doesn't match. If the old languages degrade, lower the LR, add more replay or use a smaller rank.
- The published checkpoint replaces only `gpt.pth`. All other weights come from IndexTeam/IndexTTS-2.5.

```
itts25ft/   env, lang, textfront, extractors, data, modeling, losses, utils
scripts/    pipeline steps 0-5      tests/smoke_test.py      configs/turkish.yaml
webui_ft.py, mcp_server.py
```

---

## Türkçe

[IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5)'e **yeni bir dil** ekler (örnek ayar: Türkçe). Bunu yaparken mevcut dilleri (zh, en, ja, es, ar), diller arası ses klonlamayı ve duygu kontrolünü bozmaz. Upstream repoyu değiştirmez, yalnızca içe aktarır (import eder).

### Bu sürümde neler değişti

- **Eğitilmemiş metin satırları eğitiliyor (`--train-text-rows`).** Taban model, Türkçenin kullandığı tiktoken satırlarını hiç eğitmemişti (` kaç`, ` çok`, `ı`, `ş`, `<|tr|>` için norm ~0,1, eğitilmiş satırlarda 1–3). Bu yüzden kısa kelimeler birbirine karışıyordu (` kaç` → ` evet`, ` tekrar` → ` belki`). Artık yalnızca bu satırlar eğitiliyor: gradyan maskesiyle, ayrı öğrenme oranı grubunda, weight decay olmadan ve eğitilmiş alt parçalardan başlatılarak. `configs/turkish.yaml` içinde açık.
- **Veri kalitesi:** `prepare_manifest.py`, ASR ile yazıya dökülmüş veri için filtreler içeriyor (`--max-duration`, `--min-cps/--max-cps`, `--max-same-text`, `--exclude-regex`). `preprocess.py` uzun klipleri artık **atıyor**. Eski sürüm sesi kesip transkripti tam bırakıyordu, bu da modele kelime atlamayı öğretiyordu.
- **Türkçe normalizasyon:** `%1,5,` ve `1.234,50` gibi sayılar, sıra sayısı ile cümle sonu ayrımı (`25. Yani`), bağlama göre kısaltmalar (`yak. 5 km`, `Av. Mehmet`).
- **Eğitim:** Eğitilen metin satırlarının listesi `text_rows.json` dosyasına yazılıyor ve eğitime kaldığı yerden devam edilince aynı kalıyor. Ayar, ~490 saatlik veri için 4 epoch.
- **WebUI** (`webui_ft.py`): eğitim checkpoint'lerini export etmeden yükler; token önizlemesinde eğitilen ve eğitilmemiş satırlar görünür.
- **MCP sunucusu** (`mcp_server.py`): modeli Claude Desktop, Cursor, VS Code ve diğer MCP istemcilerinden kullanmayı sağlar.
- Daha fazla smoke test (`tests/smoke_test.py`).

Türkçe eğitimimizdeki sonuç (490 saat): eğitimde görülmemiş konuşmacılarda kelime hata oranı (WER) adım 8000'de %7,1, adım 12000'de %6,3. Kelime karışması giderildi; diğer diller kontrol testlerini geçiyor.

### Hızlı başlangıç

```bash
pip install -r requirements.txt                              # index-tts-2.5 ortamında
python scripts/check_setup.py --lang tr --load-model        # 0. repo, ağırlık ve dil slotunu doğrula
python scripts/prepare_manifest.py ...                      # 1. manifest (+ kalite filtreleri)
python scripts/preprocess.py ...                            # 2. özellik önbelleği (.npz)
python scripts/build_pairs.py ...                           # 3. prompt/hedef çiftleri (aynı konuşmacı, farklı klip)
python scripts/train.py --config configs/turkish.yaml       # 4. eğitim (LoRA + dil satırı + metin satırları)
python scripts/export.py --checkpoint <run>/checkpoints/best.pt --output gpt.pth --lang tr
python scripts/synthesize.py --gpt-checkpoint gpt.pth --lang tr --normalizer turkish --case tr_lower \
    --prompt-audio ref.wav --text "Merhaba." --output out.wav --cross-lingual-check
```

Repo kardeş klasörde otomatik bulunur; bulunamazsa `INDEXTTS25_REPO` ve `INDEXTTS25_MODEL_DIR` ortam değişkenlerini ayarla. Ön işleme ve sentezde aynı `--normalizer/--case` değerlerini kullan.

### WebUI

```bash
webui.bat                        # http://127.0.0.1:7860 (GPU 0); --gpu 1 --port 7861 --qwen-emo
```

Herhangi bir checkpoint seçebilirsin (`best.pt`, `step*.pt`, export edilmiş `.pth` ya da orijinal model). Token önizlemesinde fine-tune'da eğitilen satırlar yeşil, hâlâ eğitilmemiş satırlar kırmızı görünür.

### MCP sunucusu

Araçlar: `synthesize`, `list_voices`, `list_checkpoints`, `load_checkpoint`. Referans sesleri `voices/` klasörüne koy (dosya adı = ses adı). Çıktılar `outputs/mcp/` altına yazılır. İstemci ayarı İngilizce bölümdeki JSON ile aynı. Tek modeli birden çok istemci paylaşsın istersen `mcp_server.bat --transport http --port 8765` ile başlat ve `http://127.0.0.1:8765/mcp` adresine bağlan.

### Notlar

- **Veri:** 20 saat üzeri ve 50'den fazla konuşmacı rahat bir başlangıç. Prompt ve hedef, aynı konuşmacının farklı kayıtları olmalı.
- **Eski dilleri korumak:** dil satırı gradyan maskesi, export'ta birleştirilen LoRA, isteğe bağlı replay (`yol::en@0.15`), donuk duygu dalı.
- **Sorun giderme:** Kısa kelimeler karışıyorsa `--train-text-rows` kullan. Loss iyi ama ses bozuksa metin işleme ayarları eğitimle uyuşmuyordur. Eski diller bozulduysa öğrenme oranını düşür, replay'i artır ya da rank'i küçült.
- Yayınlanan checkpoint yalnızca `gpt.pth` dosyasını değiştirir; diğer ağırlıklar IndexTeam/IndexTTS-2.5'ten gelir.
