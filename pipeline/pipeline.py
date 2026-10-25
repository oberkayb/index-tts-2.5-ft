#!/usr/bin/env python3
"""
IndexTTS-2.5 dil eğitimi pipeline'ı (index-tts-2.5-ft scriptlerini yönetir).

    uv run --project ..\\..\\index-tts python pipeline.py <adım> [seçenekler]
    ui.bat                                    # aynı adımlar tarayıcıdan

Adımlar, sırayla:
    setup       ağırlıklar, orijinal model doğrulaması, dataset kontrolü, smoke test
    manifest    dataset.jsonl -> filtrelenmiş manifest
    preprocess  GPU feature cache (yarıda kalırsa kaldığı yerden devam eder)
    pairs       prompt/hedef çiftleri
    train       eğitim (--resume auto; --probe N ile önce VRAM/hız ölçümü)
    export      eğitim checkpoint'i -> adaptör (.safetensors) ya da --merged ile tam gpt.pth
    eval        test cümlelerini sentezle, Whisper ile WER raporu çıkar
    status      neyin hazır olduğunu göster

Bütün ayarlar pipeline.toml'da (yoksa pipeline.example.toml'dan kopyalanır).
Dataset klasörü yalnızca okunur: bir çıktı yolu oraya düşerse pipeline durur.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path
from typing import Dict, List, Optional

# GPU numaraları nvidia-smi ile aynı olsun.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "pipeline.toml"
EXAMPLE_CONFIG = HERE / "pipeline.example.toml"
STEPS = ("setup", "manifest", "preprocess", "pairs", "train", "export", "eval")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

#: IndexTTS-2.5 sürümünün dosyaları (gpt.pth ayrıca ele alınır).
MODEL_FILES = (
    "config.yaml", "codec.pth", "feat1.pt", "feat2.pt", "s2mel.pth",
    "wav2vec2bert_stats.pt", "multilingual_zh_ja_yue_char_del.tiktoken",
)
MODEL_DIRS = ("qwen0.6bemo4-merge", "hf_cache")

#: Stok 2.5'te eğitilmemiş <|tr|> dil satırının normu ~0,17; eski fine-tune ~0,64.
STOCK_LANG_ROW_MAX = 0.5


class PipelineError(RuntimeError):
    pass


class Pipeline:
    def __init__(self, config_path: Path = DEFAULT_CONFIG, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.config_path = Path(config_path).resolve()
        if not self.config_path.is_file() and self.config_path == DEFAULT_CONFIG and EXAMPLE_CONFIG.is_file():
            shutil.copy2(EXAMPLE_CONFIG, self.config_path)
            print("   pipeline.toml oluşturuldu (pipeline.example.toml'dan); yolları kendine göre düzenle.")
        with self.config_path.open("rb") as handle:
            self.cfg = tomllib.load(handle)

        paths = self.cfg["paths"]
        self.dataset_jsonl = self._path(paths["dataset_jsonl"])
        self.dataset_root = self._path(paths["dataset_root"])
        self.upstream = self._path(paths["upstream_repo"])
        self.ft = self._path(paths["ft_repo"])
        self.model_dir = self._path(paths["model_dir"])
        self.work = self._path(paths["work_dir"])
        seed = paths.get("seed_model_dir") or ""
        self.seed_model_dir = self._path(seed) if seed else None
        self.seed_gpt = paths.get("seed_gpt", "gpt.pth.stock")

        self.lang = self.cfg["run"]["lang"]
        self.run_name = self.cfg["run"]["name"]
        self.data_dir = self.work / "data"
        self.run_dir = self.work / "runs" / self.run_name
        self.logs_dir = self.work / "logs"
        self.eval_dir = self.work / "eval" / self.run_name

        for out in (self.work, self.model_dir):
            self.guard(out)

    # ------------------------------------------------------------------ #
    # paths and safety
    # ------------------------------------------------------------------ #

    def _path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.config_path.parent / path).resolve()

    def guard(self, path: Path) -> Path:
        """Dataset klasörünün içine düşen her çıktı yolunu reddet."""
        resolved = Path(path).resolve()
        for root in [self.dataset_root] + [ds["root"] for ds in self.datasets()[1:]]:
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            raise PipelineError(
                "Çıktı yolu dataset klasörünün içinde: " + str(resolved) + "\n"
                "Dataset salt okunur; work_dir / model_dir ayarını başka bir yere çevir."
            )
        return resolved

    def datasets(self) -> List[Dict[str, object]]:
        pre = self.cfg["preprocess"]
        items: List[Dict[str, object]] = [{
            "lang": self.lang, "jsonl": self.dataset_jsonl, "root": self.dataset_root,
            "weight": 1.0, "normalizer": pre["normalizer"], "case": pre["case"], "replay": False,
        }]
        for rp in self.cfg.get("replay", []):
            items.append({
                "lang": rp["lang"], "jsonl": self._path(rp["jsonl"]), "root": self._path(rp["root"]),
                "weight": float(rp.get("weight", 0.15)), "normalizer": rp.get("normalizer", "none"),
                "case": rp.get("case", "lower"), "replay": True,
            })
        return items

    def lang_dir(self, lang: str) -> Path:
        return self.data_dir / lang

    # ------------------------------------------------------------------ #
    # subprocesses
    # ------------------------------------------------------------------ #

    def env(self, gpu: Optional[object] = None) -> Dict[str, str]:
        """gpu: numara -> yalnız o GPU görünür; "" -> hiç GPU görünmez; None -> dokunma."""
        env = dict(os.environ)
        env.update({
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
            "INDEXTTS25_REPO": str(self.upstream),
            "INDEXTTS25_MODEL_DIR": str(self.model_dir),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        })
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return env

    def run(self, name: str, args: List[object], cwd: Path, gpu: Optional[object] = None,
            prefix: str = "") -> List[str]:
        """Bir Python scriptini çalıştır; çıktıyı hem ekrana hem work/logs'a yaz.

        ``prefix`` paralel çalışan süreçlerin ekrandaki satırlarını ayırt eder.
        """
        cmd = [sys.executable] + [str(a) for a in args]
        print(prefix + ">> " + subprocess.list2cmdline(cmd))
        if self.dry_run:
            print("   (dry-run: çalıştırılmadı" + ("" if gpu is None else ", GPU " + str(gpu)) + ")")
            return []
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = self.logs_dir / (stamp + "_" + name + ".log")
        print(prefix + "   log: " + str(log_path) + ("" if gpu is None else "   (GPU " + (str(gpu) or "yok") + ")"))
        tail: List[str] = []
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + subprocess.list2cmdline(cmd) + "\n\n")
            proc = subprocess.Popen(
                cmd, cwd=str(cwd), env=self.env(gpu), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    sys.stdout.write(prefix + line)
                    log.write(line)
                    log.flush()
                    tail.append(line)
                    del tail[:-300]
            except KeyboardInterrupt:
                # Alt süreç de Ctrl+C aldı; eğitim checkpoint kaydedip kendisi çıkar, onu bekle.
                print(prefix + "   (Ctrl+C: alt süreç kaydedip kapanıyor, bekleniyor...)", flush=True)
                for line in proc.stdout:
                    sys.stdout.write(prefix + line)
                    log.write(line)
                proc.wait()
                raise
            code = proc.wait()
        if code != 0:
            raise PipelineError(name + " başarısız (çıkış kodu " + str(code) + "). Log: " + str(log_path))
        return tail

    # ------------------------------------------------------------------ #
    # checks
    # ------------------------------------------------------------------ #

    def lang_row_norm(self, gpt_path: Path) -> float:
        import torch

        if str(self.upstream) not in sys.path:
            sys.path.insert(0, str(self.upstream))
        from indextts.utils.tokenizer import lang_to_token

        state = torch.load(str(gpt_path), map_location="cpu", mmap=True, weights_only=False)
        state = state.get("model", state)
        return state["lang_embedding.weight"][lang_to_token(self.lang)].float().norm().item()

    def require_stock(self) -> None:
        gpt = self.model_dir / "gpt.pth"
        if not gpt.is_file():
            raise PipelineError("Model yok: " + str(gpt) + "  -> önce 00_setup.bat")
        norm = self.lang_row_norm(gpt)
        if norm >= STOCK_LANG_ROW_MAX:
            raise PipelineError(
                str(gpt) + " stok IndexTTS-2.5 değil: <|" + self.lang + "|> satırı zaten eğitilmiş "
                "(norm {:.2f}, stokta ~0,17). Bir fine-tune üzerine eğitim başlatmamak için durdum.\n"
                "Stok modeli geri koy (seed_model_dir/{}) ya da 00_setup.bat'ı çalıştır.".format(norm, self.seed_gpt)
            )
        print("   stok model doğrulandı (<|{}|> satır normu {:.3f})".format(self.lang, norm))

    def need(self, path: Path, hint: str) -> Path:
        if not path.is_file():
            if self.dry_run:
                print("   (dry-run: henüz yok, " + hint + " üretecek: " + str(path) + ")")
                return path
            raise PipelineError("Eksik: " + str(path) + "  -> önce " + hint)
        return path

    def ensure_whisper(self) -> None:
        """Eval'in Whisper modelini önceden indir (SHA-256 doğrulamalı, GPU kullanmaz)."""
        import whisper

        name = self.cfg["eval"]["whisper_model"]
        url = whisper._MODELS.get(name)
        if url is None:
            raise PipelineError("Bilinmeyen Whisper modeli '" + name + "'. Seçenekler: "
                                + ", ".join(whisper.available_models()))
        root = os.path.join(os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")), "whisper")
        target = os.path.join(root, os.path.basename(url))
        if not os.path.isfile(target):
            print("   indiriliyor: " + name + " -> " + target)
        whisper._download(url, root, in_memory=False)
        print("   hazır: {} ({:.2f} GB)".format(target, os.path.getsize(target) / 1024 ** 3))

    @staticmethod
    def count_lines(path: Path) -> int:
        if not path.is_file():
            return 0
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())

    # ------------------------------------------------------------------ #
    # steps
    # ------------------------------------------------------------------ #

    def step_setup(self, args, extra) -> None:
        print("\n== 1/7 Kaynak kod ==")
        for path, what in ((self.upstream / "indextts" / "infer_v2_5.py", "index-tts"),
                           (self.ft / "itts25ft" / "modeling.py", "index-tts-2.5-ft")):
            if not path.is_file():
                raise PipelineError(what + " bulunamadı: " + str(path.parent.parent))
        print("   index-tts        : " + str(self.upstream))
        print("   index-tts-2.5-ft : " + str(self.ft))
        print("   python           : " + sys.executable)
        if self.upstream not in Path(sys.executable).resolve().parents:
            print("   UYARI: bu python index-tts/.venv içinden değil; .bat dosyalarıyla çalıştır.")

        print("\n== 2/7 GPU (nvidia-smi; CUDA bağlamı açılmaz, çalışan işlere dokunulmaz) ==")
        import torch

        if torch.version.cuda is None:
            raise PipelineError("torch CUDA'sız kurulmuş: " + torch.__version__)
        try:
            gpus = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20, check=True,
            ).stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError) as exc:
            raise PipelineError("nvidia-smi çalışmadı: " + str(exc))
        for line in gpus:
            print("   GPU " + line.strip())
        print("   torch {} (CUDA {})".format(torch.__version__, torch.version.cuda))
        for step in ("preprocess", "train", "eval"):
            value = self.cfg["gpu"][step]
            for index in (value if isinstance(value, list) else [value]):
                if int(index) >= len(gpus):
                    raise PipelineError("[gpu] " + step + " = " + str(value) + " ama yalnızca "
                                        + str(len(gpus)) + " GPU var.")

        print("\n== 3/7 Model ağırlıkları ==")
        self.ensure_model_dir()

        print("\n== 4/7 Stok model kontrolü ==")
        self.require_stock()

        print("\n== 5/7 Dataset (salt okunur) ==")
        self.need(self.dataset_jsonl, "pipeline.toml'da dataset_jsonl yolunu düzelt")
        records = 0
        checked = missing = 0
        with self.dataset_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                records += 1
                if checked < 200:
                    checked += 1
                    audio = json.loads(line).get("audio", "")
                    if not (self.dataset_root / audio).is_file():
                        missing += 1
        print("   {} kayıt; ilk {} kaydın {} tanesinin ses dosyası eksik".format(records, checked, missing))
        if missing:
            raise PipelineError("dataset_root yanlış olabilir: ses dosyaları bulunamadı.")

        print("\n== 6/7 Whisper modeli (06_eval için) ==")
        self.ensure_whisper()

        print("\n== 7/7 Smoke test + kurulum kontrolü (CPU) ==")
        self.run("smoke_test", [self.ft / "tests" / "smoke_test.py"], cwd=self.ft)
        self.run("check_setup", [
            self.ft / "scripts" / "check_setup.py", "--lang", self.lang,
            "--normalizer", self.cfg["preprocess"]["normalizer"], "--case", self.cfg["preprocess"]["case"],
            "--sample-text", HERE / self.cfg["eval"]["sentences"],
        ], cwd=self.ft)
        print("\nKurulum tamam. Sıradaki: 01_manifest.bat")

    def ensure_model_dir(self) -> None:
        md = self.model_dir
        missing = [f for f in MODEL_FILES + ("gpt.pth",) if not (md / f).is_file()]
        missing += [d for d in MODEL_DIRS if not (md / d).is_dir() or not any((md / d).iterdir())]
        if not missing:
            print("   " + str(md) + " hazır")
        elif self.seed_model_dir is not None and (self.seed_model_dir / "config.yaml").is_file():
            seed = self.seed_model_dir
            print("   eksikler " + str(seed) + " klasöründen kopyalanıyor: " + ", ".join(missing))
            md.mkdir(parents=True, exist_ok=True)
            for name in missing:
                src = seed / (self.seed_gpt if name == "gpt.pth" else name)
                if not src.exists():
                    raise PipelineError("Kaynakta yok: " + str(src))
                started = time.time()
                if src.is_dir():
                    shutil.copytree(src, md / name, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, md / name)
                print("     {:<45} <- {}  ({:.0f} sn)".format(name, src.name, time.time() - started))
        else:
            print("   Hugging Face'ten indiriliyor: IndexTeam/IndexTTS-2.5 -> " + str(md))
            from huggingface_hub import snapshot_download

            snapshot_download("IndexTeam/IndexTTS-2.5", local_dir=str(md))

        # w2v-bert / CAMPPlus / BigVGAN: yoksa indirir, varsa dokunmaz.
        if str(self.upstream) not in sys.path:
            sys.path.insert(0, str(self.upstream))
        from indextts.utils.model_download import ensure_models_available

        ensure_models_available(str(md))
        print("   yardımcı modeller (hf_cache) hazır")

    def step_manifest(self, args, extra) -> None:
        man = self.cfg["manifest"]
        for ds in self.datasets():
            out = self.guard(self.lang_dir(ds["lang"]) / "utterances.jsonl")
            if out.is_file() and not args.force:
                print("   atlandı (var): " + str(out) + "  [--force ile yeniden]")
                continue
            cmd: List[object] = [
                self.ft / "scripts" / "prepare_manifest.py", "--format", "jsonl",
                "--input", ds["jsonl"], "--audio-dir", ds["root"], "--absolute-paths",
                "--lang", ds["lang"], "--output", out, "--max-duration", man["max_duration"],
            ]
            if not ds["replay"]:
                cmd += ["--min-cps", man["min_cps"], "--max-cps", man["max_cps"],
                        "--max-same-text", man["max_same_text"]]
                for rx in man.get("exclude_regex", []):
                    cmd += ["--exclude-regex", rx]
            self.run("manifest_" + str(ds["lang"]), cmd, cwd=self.ft)

    def preprocess_command(self, ds, manifest: Path, output_dir: Path, val_speakers: int, device: str) -> List[object]:
        pre = self.cfg["preprocess"]
        cmd: List[object] = [
            self.ft / "scripts" / "preprocess.py", "--manifest", manifest,
            "--output-dir", output_dir, "--model-dir", self.model_dir,
            "--lang", ds["lang"], "--normalizer", ds["normalizer"], "--case", ds["case"],
            "--min-seconds", pre["min_seconds"], "--max-seconds", pre["max_seconds"],
            "--device", device, "--report-every", 500,
        ]
        if val_speakers:
            cmd += ["--val-speakers", val_speakers]
        else:
            cmd += ["--val-size", 0]
        return cmd

    def gpu_free_gb(self, gpu: int) -> float:
        return self._nvidia_smi(gpu, "memory.free") / 1024.0

    def usable_gpus(self, gpus: List[int], need_gb: float, what: str) -> List[int]:
        """Yeterli boş VRAM'i olan GPU'lar; başka işler varsa OOM'a girmemek için."""
        usable = []
        for gpu in gpus:
            free = self.gpu_free_gb(gpu)
            ok = free >= need_gb
            print("   GPU {}: {:.1f} GB boş -> {}".format(gpu, free, "kullanılacak" if ok else
                  "ATLANDI (" + what + " için en az {:.1f} GB gerekli)".format(need_gb)))
            if ok:
                usable.append(gpu)
        return usable

    def step_preprocess(self, args, extra) -> None:
        pre = self.cfg["preprocess"]
        gpus = self.cfg["gpu"]["preprocess"]
        gpus = [int(g) for g in (gpus if isinstance(gpus, list) else [gpus])]
        need_gb = float(pre.get("min_free_gb", 9.0))
        for ds in self.datasets():
            lang_dir = self.lang_dir(ds["lang"])
            manifest = self.need(lang_dir / "utterances.jsonl", "01_manifest.bat")
            processed = self.guard(lang_dir / "processed")
            if len(gpus) > 1 and not args.limit:
                self.preprocess_parallel(ds, manifest, processed, gpus, need_gb, args, extra)
                continue
            usable = self.usable_gpus(gpus[:1], need_gb, "preprocess") if not self.dry_run else gpus[:1]
            if not usable:
                raise PipelineError("Boş GPU yok; diğer işler bitince tekrar dene ya da [gpu] preprocess'i değiştir.")
            cmd = self.preprocess_command(ds, manifest, processed, pre["val_speakers"], "cuda")
            if args.limit:
                cmd += ["--limit", args.limit]
            if args.force:
                cmd += ["--overwrite"]
            self.run("preprocess_" + str(ds["lang"]), cmd + extra, cwd=self.ft, gpu=usable[0])

    def preprocess_parallel(self, ds, manifest: Path, processed: Path, gpus: List[int], need_gb: float,
                            args, extra) -> None:
        """Manifest'i parçalara böl, her GPU'da bir işçi sıradaki parçayı işlesin.

        Tek GPU'ya iki süreç sığmaz (25 sn kliplerde süreç başına ~7,6 GB tepe ölçüldü), bu
        yüzden GPU başına tek işçi. Biten parça DONE ile işaretlenir; yarıda kalınca tekrar
        çalıştırmak yalnızca kalan parçaları işler.  Sonunda feature'lar processed/features'a
        taşınır ve train/val ayrımı tüm veri üzerinde bir kez, CPU'da yapılır.
        """
        pre = self.cfg["preprocess"]
        lang = str(ds["lang"])
        chunk_root = self.guard(self.lang_dir(lang) / "chunks")
        n_chunks = int(pre.get("chunks", 24))
        stat = manifest.stat()
        signature = "{}:{}".format(stat.st_size, stat.st_mtime_ns)

        if self.dry_run:
            print("   paralel: {} parça, GPU'lar {} (dry-run)".format(n_chunks, gpus))
            self.run("preprocess_" + lang + "_c00", self.preprocess_command(
                ds, chunk_root / "c00" / "manifest.jsonl", chunk_root / "c00" / "out", 0, "cuda"), cwd=self.ft, gpu=gpus[0])
            self.run("preprocess_" + lang + "_final", self.preprocess_command(
                ds, manifest, processed, pre["val_speakers"], "cpu"), cwd=self.ft, gpu="")
            return

        if args.force:
            for path in (chunk_root, processed):
                shutil.rmtree(path, ignore_errors=True)
        meta_path = chunk_root / "source.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("source") != signature:
                raise PipelineError("Manifest parçalar oluşturulduktan sonra değişmiş (" + str(manifest)
                                    + "). Baştan bölmek için: 02_preprocess.bat --force")
            n_chunks = int(meta["chunks"])
        else:
            rows = [l for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
            for k in range(n_chunks):
                chunk_dir = chunk_root / "c{:02d}".format(k)
                chunk_dir.mkdir(parents=True, exist_ok=True)
                # Sırayla dağıtım: her parçada kısa/uzun klip karışımı benzer, süreleri yakın olur.
                (chunk_dir / "manifest.jsonl").write_text("\n".join(rows[k::n_chunks]) + "\n", encoding="utf-8")
            meta_path.write_text(json.dumps({"source": signature, "chunks": n_chunks, "records": len(rows)}),
                                 encoding="utf-8")
            print("   {} kayıt {} parçaya bölündü: {}".format(len(rows), n_chunks, chunk_root))

        pending = [k for k in range(n_chunks) if not (chunk_root / "c{:02d}".format(k) / "DONE").is_file()]
        print("   parçalar: {} bitti, {} kaldı".format(n_chunks - len(pending), len(pending)))
        if pending:
            usable = self.usable_gpus(gpus, need_gb, "preprocess")
            if not usable:
                raise PipelineError("Hiçbir GPU'da yeterli boş VRAM yok; diğer işler bitince tekrar dene.")
            queue = list(pending)
            lock = threading.Lock()
            failed: List[int] = []

            def worker(gpu: int) -> None:
                strikes = 0
                while True:
                    with lock:
                        if not queue:
                            return
                        k = queue.pop(0)
                    chunk_dir = chunk_root / "c{:02d}".format(k)
                    cmd = self.preprocess_command(ds, chunk_dir / "manifest.jsonl", chunk_dir / "out", 0, "cuda")
                    try:
                        self.run("preprocess_{}_c{:02d}".format(lang, k), cmd + extra, cwd=self.ft, gpu=gpu,
                                 prefix="[GPU{} c{:02d}] ".format(gpu, k))
                        (chunk_dir / "DONE").write_text(dt.datetime.now().isoformat(), encoding="utf-8")
                        strikes = 0
                    except PipelineError as exc:
                        with lock:
                            failed.append(k)
                        print("[GPU{}] parça c{:02d} başarısız: {}".format(gpu, k, exc))
                        strikes += 1
                        if strikes >= 2:
                            print("[GPU{}] üst üste 2 hata; bu GPU'daki işçi durdu.".format(gpu))
                            return

            threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in usable]
            for thread in threads:
                thread.start()
            while any(t.is_alive() for t in threads):
                for thread in threads:
                    thread.join(timeout=1.0)
            left = [k for k in range(n_chunks) if not (chunk_root / "c{:02d}".format(k) / "DONE").is_file()]
            if left:
                raise PipelineError("{} parça bitmedi ({}). 02_preprocess.bat'ı tekrar çalıştır; biten parçalar "
                                    "atlanır.".format(len(left), ", ".join("c{:02d}".format(k) for k in left)))

        # Bütün parçalar bitti: feature'ları tek klasörde topla.
        features = processed / "features"
        features.mkdir(parents=True, exist_ok=True)
        moved = 0
        for k in range(n_chunks):
            src_dir = chunk_root / "c{:02d}".format(k) / "out" / "features"
            if not src_dir.is_dir():
                continue
            for entry in os.scandir(src_dir):
                target = features / entry.name
                if target.exists():
                    os.remove(entry.path)
                else:
                    os.replace(entry.path, target)
                    moved += 1
        print("   {} feature dosyası {} klasörüne taşındı".format(moved, features))
        have = {entry.name[:-4] for entry in os.scandir(features) if entry.name.endswith(".npz")}
        missing = 0
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip() and json.loads(line)["id"] not in have:
                    missing += 1
        if missing:
            print("   {} kaydın feature'ı yok (GPU geçişinde hata ya da süre dışı); son geçiş bunları CPU'da "
                  "dener, sayı büyükse yavaş olur. Ayrıntı: chunks/c*/out/preprocess_failures.jsonl".format(missing))

        # Son geçiş CPU'da: her şey önbellekte olduğu için yalnızca uzunlukları okur ve
        # konuşmacıya göre train/val ayırır.  GPU gizli, yani çalışan işlere dokunmaz.
        self.run("preprocess_" + lang + "_final", self.preprocess_command(
            ds, manifest, processed, pre["val_speakers"], "cpu") + extra, cwd=self.ft, gpu="")

    def step_pairs(self, args, extra) -> None:
        per_target = self.cfg["pairs"]["pairs_per_target"]
        for ds in self.datasets():
            lang_dir = self.lang_dir(ds["lang"])
            for split, n in (("train", per_target), ("val", 1)):
                src = self.need(lang_dir / "processed" / ("utterances_" + split + ".jsonl"), "02_preprocess.bat")
                self.run("pairs_" + str(ds["lang"]) + "_" + split, [
                    self.ft / "scripts" / "build_pairs.py", "--manifest", src,
                    "--output", self.guard(lang_dir / ("pairs_" + split + ".jsonl")), "--pairs-per-target", n,
                ], cwd=self.ft)

    def train_gpus(self) -> List[int]:
        value = self.cfg["gpu"]["train"]
        return [int(g) for g in (value if isinstance(value, list) else [value])]

    def train_command(self, output_dir: Path, train_specs: List[str], val_specs: List[str]) -> List[object]:
        """Tek GPU: index-tts-2.5-ft/scripts/train.py. Birden çok GPU: train_multi_gpu.py (aynı argümanlar)."""
        tr = self.cfg["train"]
        gpus = self.train_gpus()
        if len(gpus) > 1:
            accum = [int(a) for a in tr.get("rank_accum", [])]
            if accum and (len(accum) != len(gpus) or sum(accum) != int(tr["grad_accumulation"])):
                raise PipelineError("[train] rank_accum {} GPU sayısıyla ({}) eşleşmeli ve toplamı grad_accumulation "
                                    "({}) olmalı.".format(accum, len(gpus), tr["grad_accumulation"]))
            cmd: List[object] = [HERE / "train_multi_gpu.py", "--ft-repo", self.ft, "--gpus", ",".join(map(str, gpus))]
            if accum:
                cmd += ["--rank-accum", ",".join(map(str, accum))]
            if int(tr.get("early_stop_patience", 0)):
                cmd += ["--early-stop-patience", int(tr["early_stop_patience"])]
        else:
            cmd = [self.ft / "scripts" / "train.py"]
        pre = self.cfg["preprocess"]
        cmd += [
            "--config", self.ft / tr["config"],
            "--model-dir", self.model_dir, "--lang", self.lang, "--output-dir", output_dir,
            "--batch-size", tr["batch_size"], "--grad-accumulation", tr["grad_accumulation"],
            "--epochs", tr["epochs"], "--save-interval", tr["save_interval"],
            "--keep-checkpoints", tr["keep_checkpoints"], "--num-workers", tr["num_workers"],
            "--normalizer", pre["normalizer"], "--case", pre["case"],
            "--final-export", self.export_format(),
        ]
        lora = self.cfg.get("lora", {})
        if lora:
            cmd += ["--trainable-mode", lora.get("mode", "lora"), "--lora-rank", int(lora.get("rank", 32)),
                    "--lora-alpha", float(lora.get("alpha", 64.0)), "--learning-rate", float(lora.get("learning_rate", 1e-4))]
            if not lora.get("train_heads", True):
                cmd += ["--no-train-heads"]
        for spec in train_specs:
            cmd += ["--train-manifest", spec]
        for spec in val_specs:
            cmd += ["--val-manifest", spec]
        return cmd + list(tr.get("extra_args", []))

    def train_run_gpu(self) -> Optional[int]:
        """Tek GPU'da yalnız o GPU görünür; çok GPU'da hepsi görünür (numaralar nvidia-smi sırasıyla)."""
        gpus = self.train_gpus()
        return gpus[0] if len(gpus) == 1 else None

    def step_train(self, args, extra) -> None:
        self.require_stock()
        train_specs, val_specs = [], []
        for ds in self.datasets():
            lang_dir = self.lang_dir(ds["lang"])
            pairs = self.need(lang_dir / "pairs_train.jsonl", "03_pairs.bat")
            weight = "@" + str(ds["weight"]) if ds["replay"] else ""
            train_specs.append(str(pairs) + "::" + str(ds["lang"]) + weight)
            val = lang_dir / "pairs_val.jsonl"
            if val.is_file():
                val_specs.append(str(val) + "::" + str(ds["lang"]))

        tr = self.cfg["train"]
        gpus = self.train_gpus()
        if not self.dry_run:
            need = [float(tr.get("min_free_gb", 12.0))] + [float(tr.get("min_free_gb_aux", 9.0))] * (len(gpus) - 1)
            for gpu, gb in zip(gpus, need):
                if not self.usable_gpus([gpu], gb, "eğitim"):
                    raise PipelineError("GPU {} için yeterli boş VRAM yok; diğer işler bitince tekrar dene.".format(gpu))
        if not val_specs:
            # Boş bırakılırsa train.py config'teki varsayılan val yollarını kullanır (burada yoklar).
            raise PipelineError("Doğrulama çiftleri yok (pairs_val.jsonl) -> önce 03_pairs.bat")
        if args.probe:
            self.probe(args.probe, train_specs, val_specs, extra)
            return

        cmd = self.train_command(self.guard(self.run_dir), train_specs, val_specs) + ["--resume", "auto"]
        print("   çıktı: " + str(self.run_dir) + "   GPU: " + ",".join(map(str, gpus))
              + "   (durdurup tekrar çalıştırırsan kaldığı yerden devam eder)")
        self.run("train", cmd + extra, cwd=self.ft, gpu=self.train_run_gpu())

    def probe(self, steps: int, train_specs: List[str], val_specs: List[str], extra: List[str]) -> None:
        """En uzun çiftlerle kısa bir eğitim: tepe VRAM ve hız ölçümü."""
        tr = self.cfg["train"]
        gpus = self.train_gpus()
        probe_dir = self.guard(self.work / "runs" / (self.run_name + "_probe"))
        probe_args: List[object] = ["--max-steps", steps, "--save-interval", 0, "--log-interval", 1,
                                    "--val-interval", 0, "--final-export", "none"]
        if self.dry_run:
            cmd = self.train_command(probe_dir, [str(probe_dir / "longest_pairs.jsonl") + "::" + self.lang],
                                     [str(probe_dir / "probe_val.jsonl") + "::" + self.lang])
            self.run("train_probe", cmd + probe_args + extra, cwd=self.ft, gpu=self.train_run_gpu())
            return
        shutil.rmtree(probe_dir, ignore_errors=True)
        probe_dir.mkdir(parents=True)

        pairs_path = Path(train_specs[0].rsplit("::", 1)[0])
        rows = [json.loads(l) for l in pairs_path.open(encoding="utf-8") if l.strip()]
        rows.sort(key=lambda r: -int(r["code_len"]))
        needed = tr["batch_size"] * tr["grad_accumulation"] * (steps + 1)
        with (probe_dir / "longest_pairs.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows[:needed]:
                for key in ("target_features", "prompt_features"):
                    row[key] = str((pairs_path.parent / row[key]).resolve())
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        val_path = Path(val_specs[0].rsplit("::", 1)[0])
        with val_path.open(encoding="utf-8") as src, (probe_dir / "probe_val.jsonl").open("w", encoding="utf-8") as dst:
            for line in list(src)[:16]:
                row = json.loads(line)
                for key in ("target_features", "prompt_features"):
                    row[key] = str((val_path.parent / row[key]).resolve())
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
        longest = int(rows[0]["code_len"]) / 25.0 if rows else 0.0
        print("   probe: en uzun {} çift (en uzun ~{:.1f} sn), {} adım, GPU {}".format(
            min(needed, len(rows)), longest, steps, gpus))

        baseline = {g: self.gpu_memory_used(g) for g in gpus}
        peak = dict(baseline)
        stop = threading.Event()

        def watch() -> None:
            while not stop.is_set():
                for g in gpus:
                    peak[g] = max(peak[g], self.gpu_memory_used(g))
                stop.wait(1.0)

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            cmd = self.train_command(probe_dir, [str(probe_dir / "longest_pairs.jsonl") + "::" + self.lang],
                                     [str(probe_dir / "probe_val.jsonl") + "::" + self.lang])
            tail = self.run("train_probe", cmd + probe_args + extra, cwd=self.ft, gpu=self.train_run_gpu())
        finally:
            stop.set()
            watcher.join()
            shutil.rmtree(probe_dir, ignore_errors=True)

        rates = [float(m.group(1)) for m in (re.search(r"\| ([\d.]+) it/s", l) for l in tail) if m]
        print("\n== Probe sonucu (en kötü durum: en uzun klipler) ==")
        for g in gpus:
            total_gb = self.gpu_memory_total(g)
            print("   GPU {} tepe VRAM: {:.1f} / {:.1f} GB  (eğitim öncesi {:.1f} GB kullanımdaydı)".format(
                g, peak[g] / 1024, total_gb / 1024, baseline[g] / 1024))
            if total_gb and peak[g] > 0.92 * total_gb:
                print("   DİKKAT: GPU {} VRAM sınırına çok yakın. batch_size'ı düşür (grad_accumulation'ı artır).".format(g))
        if rates:
            rate = rates[-1]
            pairs = self.count_lines(pairs_path)
            per_epoch = pairs // (tr["batch_size"] * tr["grad_accumulation"])
            hours = per_epoch * tr["epochs"] / rate / 3600
            print("   hız: {:.2f} adım/sn -> {} adım/epoch x {} epoch ≈ en fazla {:.0f} saat".format(
                rate, per_epoch, tr["epochs"], hours))

    @staticmethod
    def gpu_memory_used(gpu: int) -> float:
        return Pipeline._nvidia_smi(gpu, "memory.used")

    @staticmethod
    def gpu_memory_total(gpu: int) -> float:
        return Pipeline._nvidia_smi(gpu, "memory.total")

    @staticmethod
    def _nvidia_smi(gpu: int, field: str) -> float:
        try:
            out = subprocess.run(
                ["nvidia-smi", "-i", str(gpu), "--query-gpu=" + field, "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            return float(out.stdout.strip().splitlines()[0])
        except Exception:  # noqa: BLE001 - only used for reporting
            return 0.0

    def checkpoints(self) -> List[Path]:
        found = []
        for path in (self.run_dir / "checkpoints").glob("step*.pt"):
            match = re.fullmatch(r"step(\d+)\.pt", path.name)
            if match:
                found.append((int(match.group(1)), path))
        return [p for _, p in sorted(found)]

    def export_format(self) -> str:
        fmt = str(self.cfg.get("export", {}).get("format", "adapter"))
        if fmt not in ("adapter", "merged"):
            raise PipelineError("[export] format 'adapter' ya da 'merged' olmalı, '" + fmt + "' verildi.")
        return fmt

    def adapters(self) -> List[Path]:
        folder = self.run_dir / "adapters"
        return sorted(folder.glob("*.safetensors"), key=lambda p: p.stat().st_mtime) if folder.is_dir() else []

    def step_export(self, args, extra) -> None:
        ckpts = self.checkpoints()
        if not ckpts and not args.best:
            raise PipelineError("Checkpoint yok: " + str(self.run_dir / "checkpoints") + "  -> önce 04_train.bat")
        if args.best:
            chosen = self.need(self.run_dir / "checkpoints" / "best.pt", "04_train.bat (ilk doğrulamadan sonra oluşur)")
        elif args.step:
            chosen = self.run_dir / "checkpoints" / ("step" + str(args.step) + ".pt")
            if not chosen.is_file():
                raise PipelineError("step" + str(args.step) + ".pt yok. Mevcut: "
                                    + ", ".join(p.stem for p in ckpts))
        else:
            chosen = ckpts[-1]
        self.require_stock()
        pre = self.cfg["preprocess"]
        if args.merged or self.export_format() == "merged":
            out = self.guard(self.run_dir / "exported" / ("gpt_" + chosen.stem + ".pth"))
            if out.is_file() and not args.force:
                print("   zaten var: " + str(out))
                return
            self.run("export_" + chosen.stem, [
                self.ft / "scripts" / "export.py", "--checkpoint", chosen, "--output", out,
                "--lang", self.lang, "--model-dir", self.model_dir,
            ], cwd=self.ft)
            print("\nTam çıkarım ağırlığı (birleştirilmiş): " + str(out))
            print("Ölçmek için: 06_eval.bat --checkpoint \"" + str(out) + "\"")
            return

        out = self.guard(self.run_dir / "adapters" / "{}_{}.safetensors".format(self.run_name, chosen.stem))
        if not (out.is_file() and not args.force):
            self.run("export_" + chosen.stem, [
                self.ft / "scripts" / "extract_adapter.py", "--checkpoint", chosen, "--out", out,
                "--base", self.model_dir / "gpt.pth", "--lang", self.lang,
                "--normalizer", pre["normalizer"], "--case", pre["case"],
            ], cwd=self.ft)
        else:
            print("   zaten var: " + str(out))
        # adapter_webui bu klasörü okur
        shared = self.ft / "adapters" / out.name
        if not self.dry_run:
            shared.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out, shared)
        print("\nAdaptör: " + str(out))
        print("Arayüzde kullanmak için kopyalandı: " + str(shared))
        print("Ölçmek için: 06_eval.bat --checkpoint \"" + str(out) + "\"")

    def step_eval(self, args, extra) -> None:
        if args.checkpoint:
            ckpt = Path(args.checkpoint).expanduser().resolve()
        elif args.step:
            ckpt = self.run_dir / "adapters" / "{}_step{}.safetensors".format(self.run_name, args.step)
            if not ckpt.is_file():
                ckpt = self.run_dir / "exported" / ("gpt_step" + str(args.step) + ".pth")
        else:
            exported = self.adapters() + (sorted((self.run_dir / "exported").glob("*.pth"))
                                          if (self.run_dir / "exported").is_dir() else [])
            exported.sort(key=lambda p: p.stat().st_mtime)
            if not exported:
                raise PipelineError("Dışa aktarılmış adaptör ya da ağırlık yok -> önce 05_export.bat (ya da --checkpoint ver)")
            ckpt = exported[-1]
        self.need(ckpt, "05_export.bat")
        gpu = int(self.cfg["gpu"]["eval"])
        if not self.dry_run and not self.usable_gpus([gpu], float(self.cfg["eval"].get("min_free_gb", 9.0)), "eval"):
            raise PipelineError("Eval GPU'sunda yeterli boş VRAM yok. Bekle ya da pipeline.toml'da [gpu] eval'i değiştir.")
        cmd: List[object] = [HERE / "evaluate.py", "--config", self.config_path, "--checkpoint", ckpt]
        if args.label:
            cmd += ["--label", args.label]
        if args.limit:
            cmd += ["--limit", args.limit]
        if args.force:
            cmd += ["--force"]
        self.run("eval", cmd + extra, cwd=HERE, gpu=gpu)

    @property
    def running_file(self) -> Path:
        return self.work / "running.json"

    def mark_running(self, step: str) -> Path:
        """Çalışan adımı work/running.json'a yaz; arayüz dışarıdan başlatılan adımları da böyle görür."""
        self.work.mkdir(parents=True, exist_ok=True)
        self.running_file.write_text(json.dumps({
            "step": step, "pid": os.getpid(), "started": time.time(), "run": self.run_name,
            "run_dir": str(self.run_dir),
        }), encoding="utf-8")
        return self.running_file

    def snapshot(self) -> Dict[str, object]:
        """Arayüz için: her adımın durumu, yapılandırılmış olarak (GPU'ya ve modele dokunmaz)."""
        lang_dir = self.lang_dir(self.lang)
        feats = lang_dir / "processed" / "features"
        chunks = lang_dir / "chunks"
        chunk_meta = json.loads((chunks / "source.json").read_text(encoding="utf-8")) \
            if (chunks / "source.json").is_file() else None
        ckpt_dir = self.run_dir / "checkpoints"
        reports = {}
        if self.eval_dir.is_dir():
            for report in sorted(self.eval_dir.glob("*/report.json")):
                data = json.loads(report.read_text(encoding="utf-8"))
                reports[report.parent.name] = {code: {"metric": v["metric"], "error_rate": v["error_rate"]}
                                               for code, v in data.get("languages", {}).items()}
        model_ok = all((self.model_dir / f).is_file() for f in MODEL_FILES + ("gpt.pth",))
        return {
            "run": self.run_name, "lang": self.lang, "config": str(self.config_path),
            "paths": {"dataset": str(self.dataset_jsonl), "dataset_exists": self.dataset_jsonl.is_file(),
                      "model_dir": str(self.model_dir), "work": str(self.work), "run_dir": str(self.run_dir)},
            "setup": {"model_files": model_ok},
            "manifest": {"records": self.count_lines(lang_dir / "utterances.jsonl")},
            "preprocess": {
                "features": sum(1 for _ in os.scandir(feats)) if feats.is_dir() else 0,
                "chunks_done": sum(1 for _ in chunks.glob("c*/DONE")) if chunks.is_dir() else 0,
                "chunks": chunk_meta["chunks"] if chunk_meta else 0,
                "train": self.count_lines(lang_dir / "processed" / "utterances_train.jsonl"),
                "val": self.count_lines(lang_dir / "processed" / "utterances_val.jsonl"),
            },
            "pairs": {"train": self.count_lines(lang_dir / "pairs_train.jsonl"),
                      "val": self.count_lines(lang_dir / "pairs_val.jsonl")},
            "train": {"checkpoints": [p.stem for p in self.checkpoints()],
                      "best": (ckpt_dir / "best.pt").is_file()},
            "export": {"adapters": [p.name for p in self.adapters()],
                       "merged": sorted(p.name for p in (self.run_dir / "exported").glob("*.pth"))
                       if (self.run_dir / "exported").is_dir() else []},
            "eval": reports,
        }

    def step_status(self, args, extra) -> None:
        print("dataset       : " + str(self.dataset_jsonl))
        for ds in self.datasets():
            lang_dir = self.lang_dir(ds["lang"])
            feats = lang_dir / "processed" / "features"
            n_feats = sum(1 for _ in os.scandir(feats)) if feats.is_dir() else 0
            print("[{}]".format(ds["lang"]))
            print("  manifest    : {:>8} kayıt".format(self.count_lines(lang_dir / "utterances.jsonl")))
            chunks = lang_dir / "chunks"
            if (chunks / "source.json").is_file():
                meta = json.loads((chunks / "source.json").read_text(encoding="utf-8"))
                done = sum(1 for d in chunks.glob("c*/DONE"))
                encoded = sum(sum(1 for _ in os.scandir(d)) for d in chunks.glob("c*/out/features") if d.is_dir())
                merged = lang_dir / "processed" / "features"
                encoded += sum(1 for _ in os.scandir(merged)) if merged.is_dir() else 0
                print("  kodlama     : {:>8} / {} klip, {} / {} parça bitti".format(
                    encoded, meta.get("records", "?"), done, meta["chunks"]))
            print("  features    : {:>8} .npz".format(n_feats))
            print("  utterances  : {:>8} train / {} val".format(
                self.count_lines(lang_dir / "processed" / "utterances_train.jsonl"),
                self.count_lines(lang_dir / "processed" / "utterances_val.jsonl")))
            print("  pairs       : {:>8} train / {} val".format(
                self.count_lines(lang_dir / "pairs_train.jsonl"), self.count_lines(lang_dir / "pairs_val.jsonl")))
        print("run           : " + str(self.run_dir))
        print("  checkpoints : " + (", ".join(p.stem for p in self.checkpoints()) or "-")
              + ("  + best.pt" if (self.run_dir / "checkpoints" / "best.pt").is_file() else ""))
        train_logs = sorted(self.logs_dir.glob("*_train.log")) if self.logs_dir.is_dir() else []
        if train_logs:
            lines = train_logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()
            last = {"step": None, "val": None, "best": None, "bad": None, "end": None}
            for line in lines:
                if line.startswith("step "):
                    last["step"] = line.split("| eta")[0].split("| hesap")[0].strip()
                elif line.startswith("  val @"):
                    last["val"] = line.strip()
                elif "en iyi doğrulama" in line:
                    last["best"] = line.strip()
                    last["bad"] = None
                elif "doğrulama iyileşmedi" in line:
                    last["bad"] = line.strip()
                elif "ERKEN DURDURMA" in line or "Durduruldu" in line or "son doğrulama" in line:
                    last["end"] = line.strip()
            print("  son log     : " + train_logs[-1].name)
            for key, label in (("step", "son adım"), ("val", "son val"), ("best", "en iyi val"),
                               ("bad", "erken durma"), ("end", "durum")):
                if last[key]:
                    print("  {:<12}: {}".format(label, last[key][:160]))
        exported = sorted((self.run_dir / "exported").glob("*.pth")) if (self.run_dir / "exported").is_dir() else []
        print("  adapters    : " + (", ".join(p.name for p in self.adapters()) or "-"))
        print("  exported    : " + (", ".join(p.name for p in exported) or "-"))
        reports = sorted(self.eval_dir.glob("*/report.json")) if self.eval_dir.is_dir() else []
        for report in reports:
            data = json.loads(report.read_text(encoding="utf-8"))
            langs = data.get("languages", {})
            print("  eval {:<22}: ".format(report.parent.name) + "  ".join(
                "{} {} {:.1%}".format(code, v["metric"], v["error_rate"]) for code, v in langs.items()))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="IndexTTS-2.5 Türkçe eğitim pipeline'ı")
    parser.add_argument("command", choices=("setup", "manifest", "preprocess", "pairs", "train",
                                            "export", "eval", "status"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true", help="Var olan çıktıyı yeniden üret.")
    parser.add_argument("--limit", type=int, default=0, help="preprocess/eval: ilk N kayıt (deneme için).")
    parser.add_argument("--probe", type=int, default=0, help="train: en uzun kliplerle N adımlık VRAM/hız ölçümü.")
    parser.add_argument("--step", type=int, default=0, help="export/eval: adım numarası.")
    parser.add_argument("--checkpoint", default=None, help="eval: ölçülecek gpt.pth yolu.")
    parser.add_argument("--best", action="store_true", help="export: doğrulama kaybı en iyi checkpoint (best.pt).")
    parser.add_argument("--merged", action="store_true", help="export: adaptör yerine tam (birleştirilmiş) gpt.pth yaz.")
    parser.add_argument("--label", default=None, help="eval: rapor klasörünün adı.")
    parser.add_argument("--dry-run", action="store_true", help="Komutları göster, çalıştırma.")
    args, extra = parser.parse_known_args(argv)

    import signal

    if hasattr(signal, "SIGBREAK"):
        # Arayüzün "Durdur"u Ctrl+Break gönderir: eğitim kaydedip kapanırken çıktısını okumaya devam et.
        signal.signal(signal.SIGBREAK, lambda *_: print("\n   (durdurma isteği: alt süreç kaydedip kapanıyor...)", flush=True))

    marker = None
    try:
        pipeline = Pipeline(args.config, dry_run=args.dry_run)
        if args.command != "status" and not args.dry_run:
            marker = pipeline.mark_running(args.command)
        getattr(pipeline, "step_" + args.command)(args, extra)
    except PipelineError as exc:
        print("\n[HATA] " + str(exc))
        return 1
    except KeyboardInterrupt:
        print("\nDurduruldu. Aynı adımı tekrar çalıştırınca kaldığı yerden devam eder.")
        return 130
    finally:
        try:
            if marker is not None and json.loads(marker.read_text(encoding="utf-8"))["pid"] == os.getpid():
                marker.unlink()
        except (OSError, ValueError, KeyError):
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
