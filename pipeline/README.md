# Eğitim pipeline'ı

Bir dataset'ten IndexTTS-2.5 için dil adaptörü üreten adımlar. Bütün ayarlar
`pipeline.toml`'da, bütün çıktılar `work/` altında. Adımları bir web arayüzünden
başlatıp izleyebilir ya da komut satırından tek tek çalıştırabilirsin.

[Türkçe](#türkçe) · [English](#english)

---

## Türkçe

### Kurulum

Ana README'deki "Ortam" bölümüne göre `index-tts` ve bu repo yan yana kurulu olmalı:

```
index-tts/            upstream kod, .venv ve checkpoints/ (orijinal IndexTTS-2.5)
index-tts-2.5-ft/
  pipeline/           bu klasör
```

Dataset bir `jsonl` dosyası: her satırda `audio` (dataset köküne göre yol), `text` ve
`speaker` alanları olmalı. Veriyi bulmak ve hazırlamak sana kalmış; pipeline dataset
klasörüne hiçbir şey yazmaz.

### Arayüz

```bash
cd index-tts-2.5-ft/pipeline
uv run --project ../../index-tts python ui/server.py
```

Windows'ta `ui.bat` aynı şeyi yapar. Tarayıcıda http://127.0.0.1:7870 açılır.

- **Adımlar** soldan sağa dizilir. Yeşil biten, sarı çalışan, kırmızı hata veren adımı gösterir.
  Bir adıma tıklayıp seçeneklerini işaretle ve başlat. Aynı anda tek adım çalışır.
- **Durdur** eğitime kaydedip kapanmasını söyler; yeniden başlatınca son kayıttan devam eder.
  İkinci basış süreci hemen kapatır.
- **Eğitim** grafiği kaybı, yumuşatılmış eğriyi ve doğrulama ölçümlerini çizer; en iyi
  doğrulama halkayla işaretlenir. Altında çalışan adımın canlı çıktısı akar.
- **Dışarıdan başlatılan adımlar** da görünür: terminalden ya da `.bat` ile açtığın bir adım
  `work/running.json`'a kendini yazar; arayüz onu çalışıyor olarak gösterir ve logunu izler.
  Böyle bir eğitimi Durdur ile de durdurabilirsin (`work/runs/<run>/STOP` dosyası bırakır;
  eğitim adım bitince kaydedip çıkar).
- Sağ üstteki düğme açık ve karanlık tema arasında geçiş yapar.
- **Ayarlar** `pipeline.toml`'u düzenler: run adı, LoRA (biçim, rank, alpha, öğrenme hızı),
  eğitim, GPU'lar, çıktı biçimi ve yollar. Yorumlar ve düzen korunur; geçersiz bir değer
  dosyaya yazılmaz.

Arayüz yalnızca bu makineden erişilir (127.0.0.1).

### Komut satırı

```bash
uv run --project ../../index-tts python pipeline.py <adım> [seçenekler]
```

| Adım | Ne yapar |
|---|---|
| `setup` | Ortamı ve GPU'yu kontrol eder. `model_dir` boşsa IndexTTS-2.5 ağırlıklarını Hugging Face'ten indirir (`seed_model_dir` verilirse oradan kopyalar). Orijinal `gpt.pth`'yi doğrular |
| `manifest` | Dataset satırlarını süre, konuşma hızı ve metin kurallarına göre süzer |
| `preprocess` | Klipleri semantik kodlara ve özelliklere çevirir. Yarıda kalırsa kalan parçalardan devam eder |
| `pairs` | Her hedef klibe aynı konuşmacıdan farklı bir referans klip eşler |
| `train` | LoRA adaptörünü eğitir. `--probe 20` en uzun kliplerle 20 adımlık bellek ölçümü yapar |
| `export` | Checkpoint'ten adaptörü çıkarır (`--best`, `--step N`; `--merged` tam `gpt.pth` yazar) |
| `eval` | Test cümlelerini seslendirip Whisper ile hata oranını ölçer |
| `status` | Hangi adımın ne ürettiğini yazar |

Windows'ta `00_setup.bat` … `06_eval.bat` ve `status.bat` aynı komutları çalıştırır.
Her adımın tam logu `work/logs/` altına yazılır. `--dry-run` komutları çalıştırmadan gösterir.

### Çıktı: adaptör

Eğitim orijinal modeli değiştirmez. Sonuç `work/runs/<run>/adapters/` altında bir
`.safetensors` adaptördür ve `export` onu reponun `adapters/` klasörüne de kopyalar.
Adaptör çıkarımda orijinal `gpt.pth`'nin üstüne takılır ve yalnızca kendi dilindeki
çağrılarda açılır; diğer dillerin çıktısı orijinal modelle aynı kalır. Kullanımı ana
README'deki "Dil adaptörleri" bölümünde.

Yeni bir deneme için ayarlardan run adını değiştir: eğitim çıktıları `work/runs/<ad>`
altına ayrı yazılır, `work/data` paylaşılır ve ön işleme tekrarlanmaz.

### Güvenlik

- Dataset salt okunur: bir çıktı yolu dataset klasörünün içine düşerse pipeline durur.
- `train` ve `export`, `model_dir/gpt.pth`'nin orijinal IndexTTS-2.5 olduğunu doğrular.
  Eğitilmiş bir modelin üstüne eğitim başlatmaz.
- Her GPU adımı başlamadan önce boş belleği kontrol eder; yer yoksa bellek hatasına
  düşmek yerine durur. Eşikler `pipeline.toml`'daki `min_free_gb` alanlarında.

### Sorun giderme

| Durum | Çözüm |
|---|---|
| `stok IndexTTS-2.5 değil` | `model_dir/gpt.pth`'nin üstüne eğitilmiş bir model kopyalanmış. Dosyayı silip `setup`'ı çalıştır |
| Bellek yetmiyor | `batch_size`'ı düşür, `grad_accumulation`'ı artır (çarpımları aynı kalsın) |
| Eğitim yarıda kesildi | `train`'i yeniden başlat; en fazla `save_interval` adım kaybolur |
| Normalizer'ı değiştirdin | `preprocess`'i "Baştan üret" (`--force`) ile çalıştır |

---

## English

The steps that turn a dataset into an IndexTTS-2.5 language adapter. All settings live in
`pipeline.toml`, all outputs under `work/`. Run and watch the steps from a web UI, or call
them one by one from the command line.

### Setup

Install `index-tts` and this repo side by side as described under "Environment" in the
main README. The dataset is a `jsonl` file with `audio` (path relative to the dataset
root), `text` and `speaker` per line. Finding and preparing the data is up to you; the
pipeline never writes into the dataset folder.

### UI

```bash
cd index-tts-2.5-ft/pipeline
uv run --project ../../index-tts python ui/server.py
```

On Windows `ui.bat` does the same. Open http://127.0.0.1:7870.

- **Steps** run left to right. Green means done, amber running, red failed. Click a step,
  tick its options and start it. One step runs at a time.
- **Stop** asks training to save and exit; starting it again resumes from that save.
  A second click kills the process at once.
- The **training** chart plots the loss, a smoothed curve and validation points, with the
  best validation circled. The live output of the running step streams below it.
- **Steps started elsewhere** show up too: a step run from a terminal or a `.bat` file writes
  itself to `work/running.json`, and the UI marks it as running and follows its log. Stop works
  for such a training run as well: it leaves a `work/runs/<run>/STOP` file, and training saves
  and exits after the current step.
- The button at the top right switches between light and dark themes.
- **Settings** edits `pipeline.toml`: run name, LoRA (mode, rank, alpha, learning rate),
  training, GPUs, output format and paths. Comments and layout stay intact; an invalid
  value is never written.

The UI listens on 127.0.0.1 only.

### Command line

```bash
uv run --project ../../index-tts python pipeline.py <step> [options]
```

Steps: `setup`, `manifest`, `preprocess`, `pairs`, `train`, `export`, `eval`, `status`
(see the Turkish table above). `setup` downloads the IndexTTS-2.5 weights from Hugging Face
when `model_dir` is empty. `--dry-run` prints the commands without running them; every step
logs to `work/logs/`.

### Output: an adapter

Training leaves the original model untouched. The result is a `.safetensors` adapter in
`work/runs/<run>/adapters/`, which `export` also copies to the repo's `adapters/` folder.
At inference it attaches to the stock `gpt.pth` and switches on only for its own language,
so the other languages sound exactly like the original model. To try another configuration,
change the run name: outputs go to `work/runs/<name>` and the preprocessed data is shared.
