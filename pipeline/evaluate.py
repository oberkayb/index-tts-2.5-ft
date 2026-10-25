#!/usr/bin/env python3
"""
Bir GPT checkpoint'ini sabit test cümleleriyle sentezle ve Whisper ile ölç.

    uv run --project ..\\index-tts python evaluate.py --checkpoint <gpt.pth> [--label ad]

Çıktı: work/eval/<run>/<label>/
    spk0/tr_000.wav ...     Türkçe test cümleleri (doğrulama konuşmacısının sesiyle)
    spk0/en.wav ...         aynı sesle en/es/zh/ja kontrolü (eski diller bozuldu mu?)
    report.json, report.md  WER/CER, en sık kelime değişimleri, hatalı cümleler

Aynı label ile tekrar çalıştırınca var olan wav'lar atlanır (--force ile yeniden).
Puanlamada referans ve Whisper çıktısı aynı Türkçe normalizer'dan geçer, böylece
"25" ile "yirmi beş" ya da "Çanakkale'de" ile "Çanakkalede" hata sayılmaz.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline import DEFAULT_CONFIG, Pipeline, PipelineError  # noqa: E402

PROBES = {
    "en": "This is a quick check that English still sounds the way it used to.",
    "es": "Esta es una prueba rápida para comprobar que el español sigue igual.",
    "zh": "这是一个简单的检查，用来确认中文没有退化。",
    "ja": "これは日本語が劣化していないかを確認する短いテストです。",
}
CHAR_LANGS = {"zh", "ja"}


# --------------------------------------------------------------------------- #
# prompts and sentences
# --------------------------------------------------------------------------- #

def pick_prompts(pl: Pipeline, count: int) -> List[Tuple[str, Path]]:
    """Doğrulama (eğitimde görülmeyen) konuşmacılarından ~10 sn'lik birer klip."""
    configured = pl.cfg["eval"].get("prompts") or []
    if configured:
        return [("ref" + str(i), pl._path(p)) for i, p in enumerate(configured)]

    lang_dir = pl.lang_dir(pl.lang)
    manifest = pl.need(lang_dir / "utterances.jsonl", "01_manifest.bat")
    audio: Dict[str, Tuple[str, float]] = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            audio[rec["id"]] = (rec["audio"], float(rec.get("duration") or 0.0))

    source = lang_dir / "processed" / "utterances_val.jsonl"
    if not source.is_file():
        source = pl.need(lang_dir / "processed" / "utterances_train.jsonl", "02_preprocess.bat")
    by_speaker: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            if rec["id"] in audio:
                by_speaker[str(rec["speaker"])].append(audio[rec["id"]])

    speakers = sorted(by_speaker, key=lambda s: (-len(by_speaker[s]), s))[:count]
    prompts = []
    for speaker in speakers:
        clips = [c for c in by_speaker[speaker] if 5.0 <= c[1] <= 15.0] or by_speaker[speaker]
        path, _ = min(clips, key=lambda c: abs(c[1] - 10.0))
        prompts.append((speaker, Path(path)))
    if not prompts:
        raise PipelineError("Prompt seçilemedi; pipeline.toml'da [eval] prompts listesini doldur.")
    return prompts


def read_sentences(path: Path, limit: int) -> List[str]:
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()]
    sentences = [l for l in lines if l and not l.startswith("#")]
    return sentences[:limit] if limit else sentences


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def make_tokenizers():
    from itts25ft.textfront import normalize_turkish, turkish_lower

    def tr_words(text: str) -> List[str]:
        text = turkish_lower(normalize_turkish(text))
        text = text.translate(str.maketrans("âîû", "aiu"))
        text = re.sub(r"['’`´]", "", text)
        return re.sub(r"[^\w\s]", " ", text).split()

    def words(text: str) -> List[str]:
        return re.sub(r"[^\w\s']", " ", text.lower()).split()

    def chars(text: str) -> List[str]:
        return list(re.sub(r"[\W_]", "", text))

    def units(lang: str, text: str) -> List[str]:
        if lang == "tr":
            return tr_words(text)
        return chars(text) if lang in CHAR_LANGS else words(text)

    return units


def align(ref: Sequence[str], hyp: Sequence[str]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Levenshtein hizalaması: ('ok'|'sub'|'del'|'ins', ref, hyp) listesi."""
    n, m = len(ref), len(hyp)
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dist[i][0] = i
    for j in range(m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist[i][j] = min(dist[i - 1][j] + 1, dist[i][j - 1] + 1,
                             dist[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


# --------------------------------------------------------------------------- #
# synthesis
# --------------------------------------------------------------------------- #

def synthesize(pl: Pipeline, ckpt: Path, out_dir: Path, prompts, sentences, force: bool) -> List[dict]:
    import torch
    from indextts.infer_v2_5 import IndexTTS2
    from indextts.utils.checkpoint import load_checkpoint

    from itts25ft.lang import resolve
    from itts25ft.textfront import TextFrontend, TextFrontendConfig
    from itts25ft.utils import set_seed

    ev = pl.cfg["eval"]
    pre = pl.cfg["preprocess"]
    gen = {k: ev[k] for k in ("do_sample", "temperature", "top_p", "top_k", "num_beams", "repetition_penalty")}
    probes = PROBES if ev.get("cross_lingual", True) else {}

    jobs = []
    for s_idx, (speaker, prompt) in enumerate(prompts):
        spk_dir = out_dir / ("spk" + str(s_idx))
        for i, raw in enumerate(sentences):
            jobs.append({"lang": "tr", "spk": "spk" + str(s_idx), "speaker": speaker, "prompt": str(prompt),
                         "idx": i, "ref": raw, "wav": str(spk_dir / ("tr_{:03d}.wav".format(i)))})
        for code, text in probes.items():
            jobs.append({"lang": code, "spk": "spk" + str(s_idx), "speaker": speaker, "prompt": str(prompt),
                         "idx": 0, "ref": text, "wav": str(spk_dir / (code + ".wav"))})

    todo = [j for j in jobs if force or not Path(j["wav"]).is_file()]
    if not todo:
        print(">> sentez atlandı: bütün wav'lar zaten var")
        return jobs

    print(">> IndexTTS-2.5 yükleniyor; checkpoint: " + str(ckpt))
    md = pl.model_dir
    log_path = out_dir / "synthesis.log"
    with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        tts = IndexTTS2(cfg_path=str(md / "config.yaml"), model_dir=str(md), use_bf16=True,
                        use_cuda_kernel=False, use_accel=False, use_qwen_emo=False)
        adapter = None
        if ckpt.suffix == ".safetensors":
            # adaptör: orijinal model olduğu gibi kalır, adaptör yalnızca kendi dilinde açılır
            from itts25ft.adapter import LanguageAdapter

            adapter = LanguageAdapter.load(ckpt).attach(tts.gpt)
        else:
            load_checkpoint(tts.gpt, str(ckpt))
        tts.gpt.eval()
    front = TextFrontend(md, resolve(pl.lang, None), TextFrontendConfig(normalizer=pre["normalizer"], case=pre["case"]))

    for n, job in enumerate(todo, 1):
        Path(job["wav"]).parent.mkdir(parents=True, exist_ok=True)
        set_seed(int(ev["seed"]) + int(job["idx"]))
        is_target = job["lang"] == pl.lang
        if adapter is not None:
            adapter.set_active(is_target)
        text = front.clean(job["ref"]) if is_target else job["ref"]
        print("   [{}/{}] {} {} {}".format(n, len(todo), job["spk"], job["lang"], job["ref"][:70]))
        with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            tts.infer(spk_audio_prompt=job["prompt"], text=text, output_path=job["wav"], lang=job["lang"],
                      text_normalization=not is_target, verbose=False, **gen)

    del tts
    gc.collect()
    torch.cuda.empty_cache()
    return jobs


def transcribe(jobs: List[dict], model_name: str) -> None:
    import librosa
    import numpy as np
    import torch
    import whisper

    print(">> Whisper '" + model_name + "' yükleniyor (ilk seferde indirilir)")
    model = whisper.load_model(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    for n, job in enumerate(jobs, 1):
        audio, _ = librosa.load(job["wav"], sr=16000, mono=True)
        result = model.transcribe(
            audio.astype(np.float32), language=job["lang"], task="transcribe", temperature=0.0,
            condition_on_previous_text=False, fp16=torch.cuda.is_available(), verbose=None,
        )
        job["hyp"] = result["text"].strip()
        if n % 10 == 0 or n == len(jobs):
            print("   transcribe {}/{}".format(n, len(jobs)))
    del model
    gc.collect()
    torch.cuda.empty_cache()


def score(jobs: List[dict]) -> dict:
    units = make_tokenizers()
    per_lang: Dict[str, Counter] = defaultdict(Counter)
    per_speaker: Dict[str, Counter] = defaultdict(Counter)
    subs, dels, ins = Counter(), Counter(), Counter()
    for job in jobs:
        ref, hyp = units(job["lang"], job["ref"]), units(job["lang"], job.get("hyp", ""))
        ops = align(ref, hyp)
        counts = Counter(op for op, _, _ in ops)
        errors = counts["sub"] + counts["del"] + counts["ins"]
        job["error_rate"] = errors / max(1, len(ref))
        job["ref_units"], job["hyp_units"] = " ".join(ref), " ".join(hyp)
        job["edits"] = [[op, r, h] for op, r, h in ops if op != "ok"]
        bucket = per_lang[job["lang"]]
        bucket.update({"ref": len(ref), "sub": counts["sub"], "del": counts["del"], "ins": counts["ins"]})
        if job["lang"] == "tr":
            per_speaker[job["spk"]].update({"ref": len(ref), "err": errors})
            for op, r, h in ops:
                if op == "sub":
                    subs[(r, h)] += 1
                elif op == "del":
                    dels[r] += 1
                elif op == "ins":
                    ins[h] += 1

    languages = {}
    for code, c in per_lang.items():
        err = c["sub"] + c["del"] + c["ins"]
        languages[code] = {
            "metric": "CER" if code in CHAR_LANGS else "WER",
            "error_rate": err / max(1, c["ref"]),
            "ref_units": c["ref"], "sub": c["sub"], "del": c["del"], "ins": c["ins"],
        }
    return {
        "languages": languages,
        "per_speaker": {s: c["err"] / max(1, c["ref"]) for s, c in sorted(per_speaker.items())},
        "top_substitutions": [[r, h, n] for (r, h), n in subs.most_common(25)],
        "top_deletions": [[w, n] for w, n in dels.most_common(15)],
        "top_insertions": [[w, n] for w, n in ins.most_common(15)],
    }


def write_report(out_dir: Path, meta: dict, summary: dict, jobs: List[dict]) -> None:
    report = dict(meta, **summary, items=jobs)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Eval: " + meta["label"], "",
             "- checkpoint: `" + meta["checkpoint"] + "`",
             "- tarih: " + meta["created"] + "  |  Whisper: " + meta["whisper"],
             "- üretim: " + json.dumps(meta["generation"]), "",
             "| dil | ölçü | hata | birim | sub | del | ins |", "|---|---|---|---|---|---|---|"]
    for code, v in summary["languages"].items():
        lines.append("| {} | {} | {:.1%} | {} | {} | {} | {} |".format(
            code, v["metric"], v["error_rate"], v["ref_units"], v["sub"], v["del"], v["ins"]))
    lines += ["", "## Konuşmacı başına TR WER", ""]
    for speaker, rate in summary["per_speaker"].items():
        prompt = next(j["prompt"] for j in jobs if j["spk"] == speaker)
        lines.append("- {}: {:.1%}  (`{}`)".format(speaker, rate, prompt))
    lines += ["", "## En sık kelime değişimleri (referans -> duyulan)", ""]
    lines += ["- {} -> {}  x{}".format(r, h, n) for r, h, n in summary["top_substitutions"]] or ["- yok"]
    if summary["top_deletions"]:
        lines += ["", "Atlanan kelimeler: " + ", ".join("{} x{}".format(w, n) for w, n in summary["top_deletions"])]
    lines += ["", "## Hatalı cümleler", ""]
    for job in sorted(jobs, key=lambda j: -j["error_rate"]):
        if job["error_rate"] == 0:
            continue
        lines += ["**{} {} {:.0%}** `{}`".format(job["spk"], job["lang"], job["error_rate"], Path(job["wav"]).name),
                  "", "- ref: " + job["ref_units"], "- hyp: " + job["hyp_units"], ""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def compare(eval_root: Path) -> None:
    reports = sorted(eval_root.glob("*/report.json"), key=lambda p: p.stat().st_mtime)
    if not reports:
        return
    print("\n== Bu koşudaki bütün eval'ler ==")
    for path in reports:
        langs = json.loads(path.read_text(encoding="utf-8")).get("languages", {})
        print("   {:<24} ".format(path.parent.name) + "  ".join(
            "{} {:.1%}".format(code, v["error_rate"]) for code, v in langs.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Checkpoint sentez + Whisper değerlendirmesi")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Ölçülecek adaptör (.safetensors) ya da gpt.pth")
    parser.add_argument("--label", default=None, help="Rapor klasörü (varsayılan: checkpoint adı)")
    parser.add_argument("--limit", type=int, default=0, help="İlk N test cümlesi")
    parser.add_argument("--force", action="store_true", help="Var olan wav'ları yeniden sentezle")
    args = parser.parse_args()

    try:
        pl = Pipeline(args.config)
        sys.path.insert(0, str(pl.ft))
        from itts25ft import env

        env.bootstrap(str(pl.upstream))
        ckpt = args.checkpoint.resolve()
        label = args.label or ckpt.stem
        out_dir = pl.guard(pl.eval_dir / label)
        out_dir.mkdir(parents=True, exist_ok=True)

        ev = pl.cfg["eval"]
        sentences = read_sentences(HERE / ev["sentences"], args.limit)
        prompts = pick_prompts(pl, int(ev["num_speakers"]))
        print(">> {} cümle x {} konuşmacı -> {}".format(len(sentences), len(prompts), out_dir))
        for speaker, prompt in prompts:
            print("   prompt " + speaker + ": " + str(prompt))

        jobs = synthesize(pl, ckpt, out_dir, prompts, sentences, args.force)
        transcribe(jobs, ev["whisper_model"])
        summary = score(jobs)
        meta = {
            "label": label, "checkpoint": str(ckpt), "created": dt.datetime.now().isoformat(timespec="seconds"),
            "whisper": ev["whisper_model"],
            "generation": {k: ev[k] for k in ("do_sample", "temperature", "top_p", "top_k", "num_beams",
                                              "repetition_penalty", "seed")},
            "prompts": [{"speaker": s, "audio": str(p)} for s, p in prompts],
        }
        write_report(out_dir, meta, summary, jobs)
    except PipelineError as exc:
        print("\n[HATA] " + str(exc))
        return 1

    print("\n== Sonuç ({}) ==".format(label))
    for code, v in summary["languages"].items():
        print("   {:<3} {} {:.1%}  ({} birim; sub {} / del {} / ins {})".format(
            code, v["metric"], v["error_rate"], v["ref_units"], v["sub"], v["del"], v["ins"]))
    if summary["top_substitutions"]:
        print("   en sık değişimler: " + ", ".join(
            "{}->{} x{}".format(r, h, n) for r, h, n in summary["top_substitutions"][:10]))
    print("   rapor: " + str(out_dir / "report.md"))
    compare(pl.eval_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
