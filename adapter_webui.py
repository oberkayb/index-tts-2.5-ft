#!/usr/bin/env python3
"""
Orijinal IndexTTS-2.5 + dil adaptörü (LoRA) ile ses üretimi.

    adapter_webui.bat                  # ya da:
    uv run --project ..\\index-tts --extra webui python adapter_webui.py [--gpu 0] [--port 7862]

Stok gpt.pth olduğu gibi yüklenir; adaptör (adapters/*.safetensors) üzerine takılır ve
yalnızca kendi dilindeki çağrılarda açılır. Diğer dillerde model birebir orijinaldir.
Adaptör üretmek için: scripts/extract_adapter.py
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

NO_ADAPTER = "Yok (orijinal IndexTTS-2.5)"
LANGS = [("Türkçe", "tr"), ("English", "en"), ("中文", "zh"), ("日本語", "ja"), ("Español", "es"), ("العربية", "ar")]
EXAMPLES = [
    "Yağmur birden bastırınca hepimiz sırılsıklam olduk!",
    "Ne demek istediğini bilmiyorum.",
    "Akşamüstü kuaföre uğrayıp saçlarımı biraz kestirmek istiyorum.",
    "Saat 14:30'da istasyonun önünde buluşalım. Fiyatlar yüzde 20 arttı.",
    "Işıklı ırmağın kıyısında ılık bir rüzgâr esiyordu.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=None, help="index-tts kaynak kodu (varsayılan: kardeş klasör)")
    p.add_argument("--model-dir", default=None, help="IndexTTS-2.5 checkpoints (stok gpt.pth)")
    p.add_argument("--adapters", action="append", default=[], help="Adaptör klasörü (varsayılan: ./adapters)")
    p.add_argument("--adapter", default=None, help="Açılışta takılacak adaptör (varsayılan: en yenisi)")
    p.add_argument("--gpu", type=int, default=0, help="GPU numarası (nvidia-smi sırası)")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7862)
    return p.parse_args()


class App:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        from itts25ft import env

        self.args = args
        repo = env.bootstrap(args.repo)
        self.model_dir = env.find_model_dir(args.model_dir, repo)
        self.device = "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
        self.folders = [Path(p) for p in args.adapters] or [HERE / "adapters"]

        from indextts.infer_v2_5 import IndexTTS2

        print(">> Orijinal IndexTTS-2.5 yükleniyor ({})".format(self.device))
        # use_accel/deepspeed ağırlıkları kopyalar; adaptör onları göremez, kapalı kalmalı
        self.tts = IndexTTS2(cfg_path=str(self.model_dir / "config.yaml"), model_dir=str(self.model_dir),
                             use_bf16=not args.fp32, device=self.device, use_cuda_kernel=False,
                             use_accel=False, use_deepspeed=False)
        self.adapter = None
        self.fronts = {}
        self.lock = threading.Lock()
        self.outputs = HERE / "outputs" / "adapter_webui"
        self.outputs.mkdir(parents=True, exist_ok=True)

    # -- adaptör ---------------------------------------------------------- #

    def choices(self):
        from itts25ft.adapter import adapter_size_mb, list_adapters

        found = sorted(list_adapters(self.folders), key=lambda p: -p.stat().st_mtime)
        return [(NO_ADAPTER, "")] + [("{}  ({:.0f} MB)".format(p.name, adapter_size_mb(p)), str(p)) for p in found]

    def use(self, path: str) -> str:
        from itts25ft.adapter import LanguageAdapter

        with self.lock:
            if self.adapter is not None:
                self.adapter.detach()
                self.adapter = None
            if not path:
                return "**Adaptör yok.** Model orijinal IndexTTS-2.5; Türkçe eğitimi yok."
            started = time.time()
            self.adapter = LanguageAdapter.load(path).attach(self.tts.gpt)
        meta = self.adapter.meta
        return "**Takılı:** `{}` · {} · {:.1f} sn\n\nMetin işleme: `{}` + `{}`. Adaptör yalnızca **{}** çağrılarında açılır.".format(
            Path(path).name, self.adapter.describe(), time.time() - started,
            meta.get("normalizer", "turkish"), meta.get("case", "tr_lower"), self.adapter.lang)

    # -- metin ------------------------------------------------------------ #

    def front(self, lang: str):
        from itts25ft.lang import resolve
        from itts25ft.textfront import TextFrontend, TextFrontendConfig

        meta = self.adapter.meta if self.adapter else {}
        key = (lang, meta.get("normalizer", "turkish"), meta.get("case", "tr_lower"))
        if key not in self.fronts:
            self.fronts[key] = TextFrontend(self.model_dir, resolve(lang, None),
                                            TextFrontendConfig(normalizer=key[1], case=key[2]))
        return self.fronts[key]

    def prepare(self, text: str, lang: str) -> Tuple[str, bool]:
        """(modele giden metin, upstream normalizasyonu açık mı)."""
        adapter_lang = self.adapter.lang if self.adapter else None
        if lang == adapter_lang or lang == "tr":
            return self.front(lang).clean(text), False
        return text, True

    # -- üretim ----------------------------------------------------------- #

    def _infer(self, prompt, text, lang, out: Path, active: bool, seed: int, emo_audio, emo_alpha, kw) -> float:
        from itts25ft.utils import set_seed

        if self.adapter is not None:
            self.adapter.set_active(active)
        if seed >= 0:
            set_seed(seed)
        model_text, normalize = self.prepare(text, lang)
        started = time.time()
        self.tts.infer(spk_audio_prompt=prompt, text=model_text, output_path=str(out), lang=lang,
                       emo_audio_prompt=emo_audio or None, emo_alpha=float(emo_alpha),
                       text_normalization=normalize, **kw)
        return time.time() - started

    def generate(self, prompt, text, lang, compare, duration_factor, emo_audio, emo_alpha,
                 temperature, top_p, top_k, num_beams, seed, max_text_tokens, interval_silence, progress=None):
        import gradio as gr
        import soundfile as sf
        import torch

        if not prompt:
            raise gr.Error("Önce bir referans ses yükle.")
        if not text or not text.strip():
            raise gr.Error("Metin boş.")
        seed = int(seed)
        if compare and seed < 0:
            seed = int(time.time()) % 100000  # karşılaştırma için iki üretim aynı tohumu kullanmalı
        kw = dict(do_sample=True, temperature=float(temperature), top_p=float(top_p),
                  top_k=int(top_k) if int(top_k) > 0 else None, num_beams=int(num_beams),
                  repetition_penalty=10.0, length_penalty=0.0, max_mel_tokens=1500,
                  max_text_tokens_per_segment=int(max_text_tokens), interval_silence=int(interval_silence),
                  duration_factor=float(duration_factor))
        stamp = time.strftime("%Y%m%d-%H%M%S")
        active = self.adapter is not None and lang == self.adapter.lang
        out_main = self.outputs / "{}_{}{}.wav".format(stamp, lang, "_adaptor" if active else "")
        out_ref: Optional[Path] = None
        with self.lock:
            self.tts.gr_progress = progress
            elapsed = self._infer(prompt, text, lang, out_main, active, seed, emo_audio, emo_alpha, kw)
            if compare and active:
                out_ref = self.outputs / "{}_{}_orijinal.wav".format(stamp, lang)
                self._infer(prompt, text, lang, out_ref, False, seed, emo_audio, emo_alpha, kw)
            if self.adapter is not None:
                self.adapter.set_active(False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        seconds = sf.info(str(out_main)).duration
        if active:
            state = "**Adaptör açık** ({})".format(self.adapter.lang)
        elif self.adapter is not None:
            state = "**Adaptör kapalı**: dil `{}`, model birebir orijinal".format(lang)
        else:
            state = "**Adaptör yok**" + (": orijinal model Türkçe eğitilmedi, hatalı okuma beklenir" if lang == "tr" else "")
        info = "{} · {:.1f} sn ses, {:.1f} sn üretim (RTF {:.2f}){}\n\n**Modele giden metin:** `{}`".format(
            state, seconds, elapsed, elapsed / max(seconds, 1e-6),
            " · tohum {}".format(seed) if seed >= 0 else "", self.prepare(text, lang)[0])
        return str(out_main), (str(out_ref) if out_ref else None), info


def build_ui(app: App):
    import gradio as gr

    choices = app.choices()
    default = app.args.adapter or (choices[1][1] if len(choices) > 1 else "")

    with gr.Blocks(title="IndexTTS-2.5 + adaptör") as demo:
        gr.HTML('<h2 style="margin-bottom:0">IndexTTS-2.5 + dil adaptörü</h2>'
                '<p style="margin-top:4px">Orijinal model değişmeden yüklenir; adaptör yalnızca kendi dilinde açılır.</p>')
        with gr.Row(equal_height=False):
            picker = gr.Dropdown(choices=choices, value=default, label="Adaptör", scale=4,
                                 info="adapters/ klasöründeki .safetensors dosyaları")
            refresh = gr.Button("Listeyi yenile", scale=1)
            attach = gr.Button("Tak", variant="primary", scale=1)
        status = gr.Markdown("**Adaptör yok.**")

        gr.Markdown("### Ses referansı")
        prompt = gr.Audio(label="Referans ses (5–15 sn, tek konuşmacı)", type="filepath", sources=["upload", "microphone"])
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                text = gr.Textbox(label="Metin", lines=4, placeholder="Seslendirilecek metni yaz")
                with gr.Row():
                    lang = gr.Dropdown(choices=LANGS, value="tr", label="Dil")
                    speed = gr.Slider(label="Hız (süre katsayısı)", minimum=0.5, maximum=2.0, value=1.0, step=0.01,
                                      info="hızlı ← 1.0 → yavaş")
                compare = gr.Checkbox(label="Aynı tohumla adaptörsüz sürümü de üret (karşılaştırma)", value=False)
                gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[text], label="Örnek cümleler")
            with gr.Column(scale=1):
                go = gr.Button("Ses üret", variant="primary")
                out_main = gr.Audio(label="Sonuç", type="filepath")
                out_ref = gr.Audio(label="Adaptörsüz (orijinal model)", type="filepath")
                info = gr.Markdown()

        with gr.Accordion("Duygu", open=False):
            emo_audio = gr.Audio(label="Duygu referans sesi (isteğe bağlı)", type="filepath", sources=["upload", "microphone"])
            emo_alpha = gr.Slider(label="Duygu ağırlığı", minimum=0.0, maximum=1.0, value=0.65, step=0.01)

        with gr.Accordion("Gelişmiş üretim ayarları", open=False):
            with gr.Row():
                temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.05)
                top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                num_beams = gr.Slider(label="num_beams", minimum=1, maximum=10, value=3, step=1)
            with gr.Row():
                seed = gr.Number(label="Tohum (-1 = rastgele)", value=-1, precision=0)
                max_text_tokens = gr.Slider(label="Segment başına en fazla metin token'ı", minimum=20, maximum=600,
                                            value=120, step=2)
                interval_silence = gr.Slider(label="Segmentler arası sessizlik (ms)", minimum=0, maximum=1000,
                                             value=200, step=10)

        def do_refresh():
            fresh = app.choices()
            return gr.update(choices=fresh)

        def do_generate(*values, progress=gr.Progress()):
            return app.generate(*values, progress=progress)

        refresh.click(do_refresh, outputs=[picker])
        attach.click(app.use, inputs=[picker], outputs=[status])
        go.click(do_generate,
                 inputs=[prompt, text, lang, compare, speed, emo_audio, emo_alpha,
                         temperature, top_p, top_k, num_beams, seed, max_text_tokens, interval_silence],
                 outputs=[out_main, out_ref, info])
        demo.load(lambda: app.use(default), outputs=[status])
    return demo


def main() -> int:
    args = parse_args()
    app = App(args)
    demo = build_ui(app)
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name=args.host, server_port=args.port, inbrowser=False, allowed_paths=[str(app.outputs)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
