# itts25ft — IndexTTS-2.5 için yeni dil fine-tune projesi

IndexTTS-2.5'e **yeni bir dil eklemek**, bunu yaparken modelin hâlihazırda
konuştuğu beş dili (zh, en, ja, es, ar), **cross-lingual ses klonlamayı** ve
**tını/duygu ayrışmasını** bozmamak için yazılmış bağımsız bir eğitim projesi.

Upstream repoyu (`index-tts-2.5/`) hiç değiştirmez; onu sadece import eder.
`git pull` yaptığınızda çakışma olmaz.

```
python scripts/check_setup.py --lang tr --load-model        # 0. doğrula
python scripts/prepare_manifest.py ...                      # 1. manifest
python scripts/preprocess.py ...                            # 2. feature cache
python scripts/build_pairs.py ...                           # 3. prompt/target çiftleri
python scripts/train.py --config configs/turkish.yaml       # 4. eğitim
python scripts/synthesize.py --cross-lingual-check ...      # 5. test
```

---

## 1. Neden yeni bir trainer? (2 → 2.5 mimari farkları)

`index-tts-2-ft` içindeki trainer IndexTTS-2 için yazılmış ve 2.5'te **çalışmaz**.
Kaynak kodda doğrulanan farklar:

| | IndexTTS-2 | IndexTTS-2.5 |
|---|---|---|
| Konuşmacı koşullama | `conformer_perceiver` → **32 latent** (mel'den) | **CAMPPlus 192-d** embedding → `spk_emb_proj: Linear(192, dim)` → **1 latent** |
| Koşullama uzunluğu | 32 + 2 = **34 pozisyon** | 1 + 2 = **3 pozisyon** |
| Dil koşullama | yok | `lang_embedding: Embedding(107, dim)`, **her text pozisyonuna eklenir** |
| Text tokenizer | SentencePiece `bpe.model` (12k) | Whisper **tiktoken** `multilingual_zh_ja_yue_char_del` |
| Dil kontrol tokenı | yok | metnin başına `<\|tr\|>` özel tokenı |
| Semantic codec | MaskGCT codec | `EnhancedCodec` (`downsample_scale=2`) |
| `speed_emb` | var | campplus modunda **yok** (2 pozisyon sıfır) |

> Not: `indextts/gpt/model_v2_5.py` dosyası ölü koddur — hiçbir yerden import
> edilmez. Gerçek model `indextts/gpt/model_v2.py` içindeki `UnifiedVoice`'un
> `spk_cond_mode="campplus"` hâlidir. `infer_v2_5.py` da onu kullanır.

Pratik sonuçları:

* Eski trainer'ın önceden hesapladığı `[32, dim]` conditioning latent'i artık
  anlamsız. Yerine **ham 192-d CAMPPlus vektörü** cache'lenmeli.
* `UnifiedVoice.forward` campplus modunda `lang_embedding`'i **uygulamıyor**
  (o fonksiyon s2mel için latent üretiyor). Eğitim forward'ını `losses.py`
  içinde `prepare_gpt_inputs` ile birebir aynı olacak şekilde yazdık.
* Türkçe için vocab ameliyatı **gerekmiyor**: `<|tr|>` zaten özel token,
  `lang_to_token("tr") = 9` ve `lang_embedding`'in 9. satırı checkpoint'te
  duruyor — sadece hiç eğitilmemiş (rastgele init). Eğittiğimiz şey tam olarak o.

---

## 2. Kurulum

```bash
# index-tts-2.5 ortamının içine kurun (torch, transformers, tiktoken, whisper
# vs. zaten orada). Bu proje sadece tensorboard/tqdm ekler.
pip install -r requirements.txt

# Repo yolunu otomatik bulur (kardeş dizin). Farklıysa:
export INDEXTTS25_REPO=/path/to/index-tts-2.5
export INDEXTTS25_MODEL_DIR=/path/to/index-tts-2.5/checkpoints
```

Model ağırlıkları indirilmemişse:

```bash
cd ../index-tts-2.5
hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints
```

Her şeyin doğru olduğunu **önce** kontrol edin — GPU saati harcamadan:

```bash
python scripts/check_setup.py --lang tr --normalizer turkish --case tr_lower \
    --sample-text data/tr_ornek_metinler.txt --load-model
```

Bu komut; repo/checkpoint sürümünü, dil slotunu, tokenizer verimliliğini
(tokens/char), token id taşması olup olmadığını ve eğiteceğiniz
`lang_embedding` satırının gerçekten boş olduğunu raporlar.

---

## 3. Veri gereksinimleri

| | Minimum | Rahat | İyi |
|---|---|---|---|
| Süre | 5 saat | 20–50 saat | 100 saat+ |
| Konuşmacı | 10 | 50+ | 200+ |
| Kayıt başına | 1–20 sn, temiz, tek konuşmacı | | |

**Konuşmacı etiketi kritiktir.** Prompt (ses referansı) ile hedef (üretilecek
cümle) *aynı konuşmacının farklı kayıtları* olmalı. Aynı kaydı hem prompt hem
hedef yapmak, modele cevabı kopyalamayı öğretir; sonuçta prompt'un içeriği
çıktıya sızar. `build_pairs.py` bunu sizin için yapar ve self-pair kalırsa uyarır.

Ses kalitesi: 16 kHz+ (24 kHz ideal), mono, normalize, uzun sessizlikler
kırpılmış. Transkriptler kelimesi kelimesine olmalı.

---

## 4. Adım adım

### 4.1 Manifest

```bash
python scripts/prepare_manifest.py \
    --format csv --input data/tr/metadata.csv --columns audio,text,speaker \
    --audio-dir data/tr/wavs --lang tr \
    --output data/tr/utterances.jsonl
```

`--format` seçenekleri: `jsonl`, `csv`, `tsv`, `ljspeech`, `folders`.
Konuşmacı sütunu yoksa `--speaker-from-path 1` ile klasör adından türetin.

### 4.2 Feature çıkarma (GPU)

```bash
python scripts/preprocess.py \
    --manifest data/tr/utterances.jsonl \
    --output-dir data/tr/processed \
    --lang tr --normalizer turkish --case tr_lower \
    --min-seconds 1.0 --max-seconds 20.0 --val-size 256
```

Her kayıt için tek bir `.npz` yazılır: `codes`, `text_ids`, `spk_emb`, `emo_vec`.
Eğitim bundan sonra hiç wav açmaz. Yeniden çalıştırmak ucuzdur — var olan
dosyalar atlanır (`--overwrite` ile zorlayabilirsiniz).

Çıkarma yolu `infer_v2_5` ile birebir aynıdır: w2v-BERT 17. katman + shipped
mean/var, `EnhancedCodec.quantize`, 80-bin Kaldi fbank → CAMPPlus, ve modelin
kendi **donuk** duygu enkoderi.

### 4.3 Prompt/target çiftleri

```bash
python scripts/build_pairs.py \
    --manifest data/tr/processed/utterances_train.jsonl \
    --output data/tr/pairs_train.jsonl --pairs-per-target 2

python scripts/build_pairs.py \
    --manifest data/tr/processed/utterances_val.jsonl \
    --output data/tr/pairs_val.jsonl --pairs-per-target 1
```

Bir konuşmacı birden fazla dilde kayıt vermişse `--cross-lingual` ekleyin:
prompt dili A, hedef dili B olan çiftler üretilir. Bu, "bu sesi klonla, şu dili
konuş" davranışının **tek doğrudan denetimidir**.

### 4.4 Eğitim

```bash
python scripts/train.py --config configs/turkish.yaml
```

veya açık bayraklarla:

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

Manifest sözdizimi: `yol[::dil[:alias]][@ağırlık]`.

### 4.5 Test

```bash
python scripts/synthesize.py \
    --gpt-checkpoint runs/tr_lora/exported/gpt.pth \
    --lang tr --normalizer turkish --case tr_lower \
    --prompt-audio ref.wav \
    --text "Merhaba, bugün hava çok güzel." \
    --output out/tr.wav \
    --cross-lingual-check
```

`--cross-lingual-check`, **aynı referans sesle** İngilizce/Çince/Japonca/İspanyolca
kısa cümleler de üretir. Bunları stok modelin çıktısıyla karşılaştırın: bozulma
varsa fine-tune temel dillere sızmış demektir.

---

## 5. Cross-lingual'i korumak (projenin asıl meselesi)

Yeni dil eklerken asıl risk **catastrophic forgetting**. Beş savunma katmanı
kurulu:

**1. Dil satırı izolasyonu.** `LanguageEmbeddingGradMask` gradyanı sadece
hedef satırda bırakır. Diğer 106 dil satırı matematiksel olarak sabit kalır —
smoke test bunu doğruluyor.

**2. Düşük ranklı güncelleme (LoRA).** Varsayılan mod. GPT gövdesi donuk;
sadece `c_attn`/`c_proj`/`c_fc` üzerine rank-32 adaptörler öğrenilir (toplam
parametrenin ~%1'i). Export sırasında adaptörler taban ağırlıklara **merge**
edilir; sonuç stok `infer_v2_5.py`'nin değişiklik olmadan yüklediği düz bir
`gpt.pth` olur.

**3. Replay.** Eski dillerden veri karıştırın (`::en@0.15`). Validation her dil
için ayrı loss raporlar; `--forgetting-guard 0.05` bir dil temel çizgisinden
sapınca ekrana uyarı basar. Bu, bozulmayı sentezden *önce* görmenizi sağlar.

**4. Donuk duygu dalı.** `emo_conditioning_encoder`, `emo_perceiver_encoder`,
`emovec_layer`, `emo_layer` ve `spk_emb_proj` varsayılan olarak dondurulur.
Tını/duygu ayrışması bu enkoderlerin özelliğidir; küçük tek dilli bir korpusla
onları yeniden eğitmek tam olarak bu özelliği kaybetme yoludur.

**5. Konuşmacı vektörü augmentasyonu.** 2.5'te tını tek bir 192-d vektörden
geçiyor; küçük bir korpusta bunu ezberlemek çok kolay. `--spk-noise-std 0.01`
gürültü ekleyerek klonlamanın genel kalmasını sağlar.

**Dürüst olmak gerekirse:** eğer elinizde aynı konuşmacının hem Türkçe hem
başka dilde kaydı yoksa, "A dilinde konuşan kişiye B dilini konuşturma"nın
doğrudan denetimi yoktur. O yetenek **zaten temel modelde var** — çünkü
konuşmacı koşullaması dilden bağımsız bir CAMPPlus vektörü ve dil ayrı bir
embedding ile veriliyor. İşimiz onu **öğretmek değil, bozmamak**. Yukarıdaki
1–5 tam olarak bunun içindir.

---

## 6. Hiperparametre rehberi

| Senaryo | mode | LR | rank | epoch |
|---|---|---|---|---|
| 5–20 saat, çok konuşmacı | `lora` | 1e-4 | 16–32 | 10–15 |
| 20–100 saat | `lora` | 1e-4 | 32–64 | 8–12 |
| 100 saat+, eski diller önemsiz | `partial` | 2e-5 | — | 5–8 |
| Tek konuşmacı / ses klonu | `lora` | 5e-5 | 8–16 | 5–10 |

* **`--lang-lr-multiplier 10`**: tek bir yeni embedding satırı, oturmuş bir
  modelin gövde LR'siyle çok yavaş öğrenir. Varsayılanı düşürmeyin.
* **`--lang-init-from`**: yeni satırı eğitilmiş bir dilden kopyalar. Latin
  alfabeli, hece zamanlamalı diller için `es` iyi bir başlangıç; vermezseniz
  rastgele init'ten başlarsınız (yavaş).
* **`--text-loss-weight 0.2`**: text CE yardımcı görev. 1.0'a çıkarmayın.
* **`--emo-source target` + `--prompt-emo-prob 0.25`**: duygu referansı
  varsayılan olarak *hedef* kayıttır (IndexTTS-2'nin ayrıştırma reçetesi);
  zamanın %25'inde prompt'unki kullanılır — çıkarımın varsayılan davranışı budur.
  İkisini birlikte eğitmek her iki kullanımı da çalışır tutar.
* **Efektif batch**: `batch_size × grad_accumulation ≥ 32` hedefleyin.
* **VRAM**: `batch_size 8` + bf16 + LoRA ≈ 24 GB. Yetmezse `--batch-size 4
  --grad-accumulation 8` ve `--max-code-tokens 1200`.

---

## 7. Sorun giderme

**"Could not locate the IndexTTS-2.5 source tree"** → `INDEXTTS25_REPO` ayarlayın
veya `--repo` verin.

**"config.yaml reports version=2.0"** → IndexTTS-2 ağırlıklarını gösteriyorsunuz.
2.5 checkpoint'i indirin.

**Token id taşması** (`check_setup` FAILED) → `checkpoints/` içindeki tiktoken
vocab 2.5'e ait değil. Ağırlıkları yeniden indirin.

**tokens/char > 0.5** → BPE dilinizi byte'lara parçalıyor. Çalışır ama daha çok
veri/adım ister. Metin normalizasyonunu (sayı/kısaltma açma) mutlaka yapın.

**Loss düşüyor ama çıktı bozuk** → neredeyse her zaman frontend uyumsuzluğu.
`preprocess.py` ile `synthesize.py`'ye **aynı** `--normalizer` ve `--case`
değerlerini verin. `synthesize.py` zaten `text_normalization=False` gönderir.

**Prompt'un içeriği çıktıya sızıyor** → self-pair'ler. `build_pairs.py`
çıktısındaki "self pairs" sayısına bakın; konuşmacı etiketlerini düzeltin.

**Eski diller bozuldu** → LR'yi düşürün, replay ağırlığını artırın, `lora-rank`'i
küçültün, `--lora-last-n-layers 8` ile sadece üst katmanları uyarlayın.

**Eğitim çöküyor / NaN** → `--amp fp16` yerine `bf16`; `--grad-clip 1.0`; LR'yi
yarıya indirin.

---

## 8. Dosya haritası

```
itts25ft/
  env.py         repo/checkpoint bulma, config yükleme, 2.5 sürüm doğrulaması
  lang.py        dil slotu çözümleme (<|xx|> tokenı + embedding satırı), alias
  textfront.py   infer_v2_5 ile birebir metin işleme + Türkçe normalizer
  extractors.py  w2v-BERT / EnhancedCodec / CAMPPlus / duygu vektörü cache'leme
  data.py        çift manifestleri, dil karışımı (replay), uzunluk bucketing
  modeling.py    model kurma, dil satırı init + gradyan maskesi, LoRA, export
  losses.py      2.5 eğitim forward'ı (lang_embedding dahil) + maskeli CE
  utils.py       seed, checkpoint rotasyonu, metrik ortalama
scripts/         0–5 adımları (yukarıda)
tests/smoke_test.py   checkpoint gerektirmeyen 30 doğrulama (saniyeler sürer)
configs/turkish.yaml  referans eğitim konfigürasyonu
```

Değişiklik yaptıysanız önce şunu çalıştırın:

```bash
python tests/smoke_test.py
```

Loss yolunu, dil satırı izolasyonunu, LoRA merge denkliğini ve export
uyumluluğunu gerçek ağırlık indirmeden doğrular.
