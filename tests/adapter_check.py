#!/usr/bin/env python3
"""
Adaptörün doğruluk testi (GPU gerekir):

  1. Kapalıyken: diğer dillerde çıktı orijinal IndexTTS-2.5 ile bit bit aynı mı?
     (takmadan önce / takılı ama kapalı / açıp kapattıktan sonra)
  2. Açıkken: Türkçe ses kodları, birleştirilmiş (merged) gpt.pth ile aynı mı?
  3. İsteğe bağlı: Whisper Türkçe çıktıyı doğru duyuyor mu?

    python tests/adapter_check.py --adapter adapters/tr_v2.safetensors --prompt ref.wav \
        [--merged checkpoints/tr/gpt.pth] [--whisper openai/whisper-large-v3-turbo]

Örneklemesiz (greedy, num_beams=1) üretir ve her üretimden önce tohumu sabitler.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
HERE = Path(__file__).resolve().parent.parent
# indextts içe aktarılınca HF önbelleğini ./checkpoints/hf_cache'e çeker; Whisper'ı varsayılan önbellekten al
DEFAULT_HF_CACHE = os.environ.get("HF_HUB_CACHE") or str(Path.home() / ".cache" / "huggingface" / "hub")
sys.path.insert(0, str(HERE))

EN = "The quick brown fox jumps over the lazy dog, and then it runs away."
TR = ["Yağmur birden bastırınca hepimiz sırılsıklam olduk!", "Ne demek istediğini bilmiyorum."]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--prompt", required=True, help="referans ses")
    p.add_argument("--merged", default=None, help="karşılaştırma için birleştirilmiş Türkçe gpt.pth")
    p.add_argument("--whisper", default=None, help="ör. openai/whisper-large-v3-turbo")
    p.add_argument("--repo", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default=str(HERE / "outputs" / "adapter_check"))
    args = p.parse_args()

    import numpy as np
    import soundfile as sf
    import torch

    from itts25ft import env
    from itts25ft.adapter import LanguageAdapter
    from itts25ft.lang import resolve
    from itts25ft.textfront import TextFrontend, TextFrontendConfig
    from itts25ft.utils import set_seed

    repo = env.bootstrap(args.repo)
    model_dir = env.find_model_dir(args.model_dir, repo)
    from indextts.infer_v2_5 import IndexTTS2
    from indextts.utils.checkpoint import load_checkpoint

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tts = IndexTTS2(cfg_path=str(model_dir / "config.yaml"), model_dir=str(model_dir), use_bf16=True,
                    device="cuda:{}".format(args.gpu), use_cuda_kernel=False, use_accel=False)
    front = TextFrontend(model_dir, resolve("tr", None), TextFrontendConfig(normalizer="turkish", case="tr_lower"))

    # GPT'nin ürettiği ses kodlarını yakala
    codes = []
    original = tts.gpt.inference_speech

    def capture(*a, **k):
        result = original(*a, **k)
        seq = result[0] if isinstance(result, tuple) else result
        codes.append(seq.detach().cpu().clone())
        return result

    tts.gpt.inference_speech = capture

    def run(text: str, lang: str, name: str):
        codes.clear()
        set_seed(1234)
        model_text, normalize = (front.clean(text), False) if lang == "tr" else (text, True)
        path = out / (name + ".wav")
        tts.infer(spk_audio_prompt=args.prompt, text=model_text, output_path=str(path), lang=lang,
                  do_sample=False, num_beams=1, text_normalization=normalize, max_mel_tokens=1500)
        wav, _ = sf.read(str(path))
        return wav, [c.clone() for c in codes]

    print("\n== 1. Kapalıyken orijinalle aynı mı (en) ==")
    en_stock, en_codes = run(EN, "en", "en_stock")
    adapter = LanguageAdapter.load(args.adapter).attach(tts.gpt)
    adapter.set_active(False)
    en_off, en_off_codes = run(EN, "en", "en_attached_off")
    adapter.set_active(True)
    adapter.set_active(False)
    en_toggled, en_tog_codes = run(EN, "en", "en_toggled_off")
    ok = True
    for label, wav, cds in (("takılı, kapalı", en_off, en_off_codes), ("aç-kapat sonrası", en_toggled, en_tog_codes)):
        same_codes = all(torch.equal(a, b) for a, b in zip(en_codes, cds)) and len(en_codes) == len(cds)
        same_wav = wav.shape == en_stock.shape and np.array_equal(wav, en_stock)
        print("  {:18s} ses kodları aynı: {} | dalga bit bit aynı: {}".format(label, same_codes, same_wav))
        ok &= same_codes and same_wav

    print("\n== 2. Açıkken Türkçe ==")
    adapter.set_active(True)
    tr_adapter = [run(t, "tr", "tr_adapter_{}".format(i)) for i, t in enumerate(TR)]
    adapter.set_active(False)
    tr_stock = [run(t, "tr", "tr_stock_{}".format(i)) for i, t in enumerate(TR)]
    if args.merged:
        adapter.detach()
        load_checkpoint(tts.gpt, args.merged)
        tts.gpt.eval().bfloat16()
        tr_merged = [run(t, "tr", "tr_merged_{}".format(i)) for i, t in enumerate(TR)]
        for i, ((_, a), (_, m)) in enumerate(zip(tr_adapter, tr_merged)):
            a, m = torch.cat([c.flatten() for c in a]), torch.cat([c.flatten() for c in m])
            n = min(len(a), len(m))
            match = (a[:n] == m[:n]).float().mean().item() if n else 0.0
            print("  cümle {}: adaptör {} kod, birleştirilmiş {} kod, ilk {} kodda eşleşme %{:.1f}".format(
                i + 1, len(a), len(m), n, 100 * match))

    if args.whisper:
        from huggingface_hub import snapshot_download
        from transformers import pipeline

        whisper_dir = snapshot_download(args.whisper, cache_dir=DEFAULT_HF_CACHE)
        asr = pipeline("automatic-speech-recognition", model=whisper_dir, device="cuda:{}".format(args.gpu),
                       torch_dtype=torch.float16)
        print("\n== 3. Whisper ne duyuyor ==")
        for i, t in enumerate(TR):
            for tag in ("adapter", "stock") + (("merged",) if args.merged else ()):
                heard = asr(str(out / "tr_{}_{}.wav".format(tag, i)), generate_kwargs={"language": "turkish"})["text"].strip()
                print("  [{:7s}] {}".format(tag, heard))
            print("  [hedef  ] {}".format(t))

    print("\nSONUÇ: kapalıyken orijinalle", "AYNI" if ok else "FARKLI", "| çıktılar:", out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
