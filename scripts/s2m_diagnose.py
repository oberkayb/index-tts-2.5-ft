#!/usr/bin/env python3
"""
S2M'yi eğitmek doğallığı artırır mı? Eğitmeden önce ölç.

IndexTTS-2.5'te GPT'den S2M'ye yalnızca ses kodları geçer. Aynı cümle üç kolda puanlanır:

  A  gerçek kayıt                                   (tavan)
  R  gerçek kaydın ses kodları -> S2M -> BigVGAN     ("GPT kusursuz olsaydı")
  G  Türkçe adaptörlü GPT'nin kodları -> S2M -> BigVGAN (bugünkü sistem)

R ve G'de referans ses, tohum, S2M ve vokoder aynıdır; fark yalnızca kodlardan gelir.
UTMOS22-strong (SpeechMOS v1.2.0) ve 8 kHz bantta Whisper ile puanlanır.

  A - R büyük  -> S2M/vokoder Türkçeyi kaybettiriyor: S2M eğitimi işe yarar
  R - G büyük  -> açık GPT'de (prozodi/kodlar)
  ikisi küçük  -> doğallık zaten tavana yakın; sınırı referans ses belirliyor

İki girdi biçimi:

  --clips diagnose/turkce/manifest.jsonl   konuşmacı klasörlerinde wav (+ metin); A, R, G
                                           referans = aynı konuşmacının başka bir kaydı
  --manifest .../utterances_val.jsonl      ses yok, preprocess .npz kodları; R, G
                                           referans = --prompt

    python scripts/s2m_diagnose.py --adapter adapters/tr_v2.safetensors --clips diagnose/turkce/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
# indextts içe aktarılınca HF önbelleğini ./checkpoints/hf_cache'e çeker; Whisper'ı varsayılan önbellekten al
DEFAULT_HF_CACHE = os.environ.get("HF_HUB_CACHE") or str(Path.home() / ".cache" / "huggingface" / "hub")
ARMS = ("A", "R", "G")


# --------------------------------------------------------------------------- #
# girdiler
# --------------------------------------------------------------------------- #

def items_from_npz(manifest: Path, n: int, prompt: str, seed: int = 0):
    """Ses yok: preprocess'in .npz kodları. Konuşmacılara eşit dağılmış n kayıt."""
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if 60 <= r["code_len"] <= 600]
    by_spk = defaultdict(list)
    for r in rows:
        by_spk[r["speaker"]].append(r)
    rng = random.Random(seed)
    for v in by_spk.values():
        rng.shuffle(v)
    out = []
    while len(out) < n and any(by_spk.values()):
        for spk in sorted(by_spk):
            if by_spk[spk] and len(out) < n:
                r = by_spk[spk].pop()
                out.append({"id": r["id"], "speaker": r["speaker"], "text": r["text"], "prompt": prompt,
                            "npz": str(manifest.parent / r["features"]), "real": None})
    return out


def items_from_clips(manifest: Path, n: int):
    """Konuşmacı klasörlerindeki kayıtlar; referans = aynı konuşmacının sıradaki kaydı."""
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_spk = defaultdict(list)
    for r in rows:
        by_spk[r["speaker"]].append(r)
    out = []
    for spk in sorted(by_spk):
        clips = by_spk[spk]
        if len(clips) < 2:
            continue
        for i, r in enumerate(clips):
            wav = Path(r["wav"])
            out.append({"id": "{}_{}".format(spk, wav.stem), "speaker": spk, "text": r["text"],
                        "prompt": clips[(i + 1) % len(clips)]["wav"], "npz": None, "real": str(wav)})
    return out[:n]


# --------------------------------------------------------------------------- #
# ölçüm yardımcıları
# --------------------------------------------------------------------------- #

_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})


def norm_words(text: str):
    text = text.translate(_TR_LOWER).lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return text.replace("'", "").split()


def word_errors(ref: str, hyp: str):
    r, h = norm_words(ref), norm_words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def mean_ci(xs):
    m = statistics.mean(xs)
    se = statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return m, 1.96 * se


# --------------------------------------------------------------------------- #
# sentez
# --------------------------------------------------------------------------- #

def synthesize(items, args, out: Path):
    import numpy as np
    import torch

    from itts25ft import env
    from itts25ft.adapter import LanguageAdapter
    from itts25ft.lang import resolve
    from itts25ft.textfront import TextFrontend, TextFrontendConfig
    from itts25ft.utils import set_seed

    repo = env.bootstrap(args.repo)
    model_dir = env.find_model_dir(args.model_dir, repo)
    from indextts.infer_v2_5 import IndexTTS2

    device = "cuda:{}".format(args.gpu)
    tts = IndexTTS2(cfg_path=str(model_dir / "config.yaml"), model_dir=str(model_dir), use_bf16=True,
                    device=device, use_cuda_kernel=False, use_accel=False)
    adapter = LanguageAdapter.load(args.adapter).attach(tts.gpt)
    adapter.set_active(True)
    meta = adapter.meta
    front = TextFrontend(model_dir, resolve("tr", None),
                         TextFrontendConfig(normalizer=meta.get("normalizer", "turkish"), case=meta.get("case", "tr_lower")))

    extractor = None
    if any(it["real"] for it in items):
        from itts25ft.extractors import SPK_SR, FeatureExtractor

        extractor = FeatureExtractor(model_dir, env.load_model_config(model_dir), device=device)

    def real_codes_of(it):
        if it["npz"]:
            return np.load(it["npz"])["codes"]
        wav, sr, _ = extractor.load_audio(it["real"])
        emb = extractor.w2v_embedding(extractor._resample(wav, sr, SPK_SR))  # preprocess ile aynı: [1, N]
        return extractor.semantic_codes(emb).squeeze(0).cpu().numpy()

    gpt_infer = tts.gpt.inference_speech
    current = {"codes": None}

    def fixed_codes(*a, **k):
        # R kolu: GPT'yi atla, gerçek kaydın kodlarını (sonuna durma kodu ekleyerek) ver
        c = torch.as_tensor(current["codes"], dtype=torch.long, device=device)[None]
        stop = torch.full((1, 1), tts.stop_mel_token, dtype=torch.long, device=device)
        return torch.cat([c, stop], dim=1), None

    started = time.time()
    for k, it in enumerate(items, 1):
        text = front.clean(it["text"])
        for arm in ("R", "G"):
            path = out / arm / (it["id"] + ".wav")
            if path.exists():
                continue
            if arm == "R":
                current["codes"] = real_codes_of(it)
                tts.gpt.inference_speech = fixed_codes
            else:
                tts.gpt.inference_speech = gpt_infer
            set_seed(args.seed)
            tts.infer(spk_audio_prompt=it["prompt"], text=text, output_path=str(path), lang="tr",
                      text_normalization=False, max_text_tokens_per_segment=600, max_mel_tokens=1800,
                      do_sample=True, top_p=0.8, top_k=30, temperature=0.8, num_beams=3,
                      repetition_penalty=10.0, length_penalty=0.0)
        if k % 10 == 0:
            print("  sentez {}/{} ({:.0f} sn)".format(k, len(items), time.time() - started), flush=True)
    tts.gpt.inference_speech = gpt_infer


# --------------------------------------------------------------------------- #
# puanlama
# --------------------------------------------------------------------------- #

def score(items, args, out: Path):
    import librosa
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download
    from transformers import pipeline

    mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval().cuda(args.gpu)
    whisper_dir = snapshot_download(args.whisper, cache_dir=DEFAULT_HF_CACHE)
    asr = pipeline("automatic-speech-recognition", model=whisper_dir, device="cuda:{}".format(args.gpu),
                   torch_dtype=torch.float16)

    def load(path):
        w, sr = sf.read(str(path), dtype="float32")
        return (w.mean(1) if w.ndim > 1 else w), sr

    def utmos(path) -> float:
        w, sr = load(path)
        with torch.no_grad():
            return mos(torch.from_numpy(librosa.resample(w, orig_sr=sr, target_sr=16000))[None].cuda(args.gpu), 16000).item()

    def hear(path) -> str:
        w, sr = load(path)
        w = librosa.resample(librosa.resample(w, orig_sr=sr, target_sr=8000), orig_sr=8000, target_sr=16000)
        return asr({"raw": w, "sampling_rate": 16000}, generate_kwargs={"language": "turkish"})["text"].strip()

    report = []
    for it in items:
        row = {"id": it["id"], "speaker": it["speaker"], "text": it["text"], "prompt": it["prompt"]}
        paths = {"A": it["real"], "R": out / "R" / (it["id"] + ".wav"), "G": out / "G" / (it["id"] + ".wav")}
        for arm, path in paths.items():
            if not path or not Path(path).exists():
                continue
            hyp = hear(path)
            errs, n = word_errors(it["text"], hyp)
            row[arm] = {"utmos": utmos(path), "hyp": hyp, "err": errs, "words": n, "dur": sf.info(str(path)).duration}
        report.append(row)
    prompts = sorted({it["prompt"] for it in items})
    prompt_mos = statistics.mean(utmos(p) for p in prompts)
    return report, prompt_mos


def summarize(report, prompt_mos, out: Path):
    arms = [a for a in ARMS if all(a in x for x in report)]
    lines = ["", "== S2M teşhisi ({} cümle, {} konuşmacı) ==".format(len(report), len({x['speaker'] for x in report})),
             "  referans seslerin UTMOS'u (ort.) : {:.2f}".format(prompt_mos)]
    names = {"A": "A gerçek kayıt          ", "R": "R gerçek kodlar -> S2M  ", "G": "G Türkçe GPT -> S2M     "}
    for arm in arms:
        m, ci = mean_ci([x[arm]["utmos"] for x in report])
        wer = 100 * sum(x[arm]["err"] for x in report) / max(1, sum(x[arm]["words"] for x in report))
        lines.append("  {}: UTMOS {:.2f} ± {:.2f}   WER %{:.1f}".format(names[arm], m, ci, wer))
    for a, b, meaning in (("A", "R", "S2M + vokoderin kaybı"), ("R", "G", "GPT'nin kaybı")):
        if a in arms and b in arms:
            d, ci = mean_ci([x[a]["utmos"] - x[b]["utmos"] for x in report])
            lines.append("  {} - {} (eşli) = {:+.2f} ± {:.2f}   ({})".format(a, b, d, ci, meaning))
    lines += ["", "  Karşılaştırma (Freya, Anka referansı): Anka 3,52 · Trendyol 3,82 · TR v2 3,17"]
    text = "\n".join(lines)
    print(text)
    (out / "report.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in report), encoding="utf-8")
    (out / "summary.txt").write_text(text, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--clips", help="konuşmacı klasörleri manifest'i (speaker, wav, text)")
    src.add_argument("--manifest", help="preprocess utterances_*.jsonl (features + text)")
    p.add_argument("--prompt", help="--manifest ile: tüm kayıtlar için referans ses")
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--whisper", default="openai/whisper-large-v3-turbo")
    p.add_argument("--repo", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default=str(HERE / "outputs" / "s2m_diagnose"))
    p.add_argument("--skip-synth", action="store_true", help="yalnızca var olan wav'ları puanla")
    args = p.parse_args()
    if args.manifest and not args.prompt:
        p.error("--manifest ile --prompt gerekli")

    items = items_from_clips(Path(args.clips), args.n) if args.clips else items_from_npz(Path(args.manifest), args.n, args.prompt)
    out = Path(args.out)
    for arm in ("R", "G"):
        (out / arm).mkdir(parents=True, exist_ok=True)
    print(">> {} kayıt, {} konuşmacı".format(len(items), len({it['speaker'] for it in items})))

    if not args.skip_synth:
        import torch

        synthesize(items, args, out)
        torch.cuda.empty_cache()
    report, prompt_mos = score(items, args, out)
    summarize(report, prompt_mos, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
