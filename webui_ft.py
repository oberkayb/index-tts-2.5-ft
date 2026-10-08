#!/usr/bin/env python3
"""
Fine-tune edilmiş IndexTTS-2.5 için WebUI (upstream webui.py'nin düzenine benzer).

    webui.bat                      # ya da:
    uv run --project ..\\index-tts --extra webui python webui_ft.py [--gpu 0] [--port 7860]

Upstream arayüzünden farkları:
  * Checkpoint seçimi: eğitim checkpoint'i (best.pt / step*.pt, export beklemeden bellekte
    birleştirilir), export edilmiş gpt.pth, stok model ya da elle girilen yol.
  * Türkçe, eğitimdeki frontend ile (turkish + tr_lower, text_normalization=False);
    diğer diller upstream normalizasyonuyla.
  * Metin/token önizleme: modele giden token'lar, fine-tune'da eğitilen satırlar ve hâlâ
    eğitilmemiş satırlar renklendirilmiş olarak (kelime karışmasını teşhis için).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LANGS = [("Türkçe (fine-tune)", "tr"), ("English", "en"), ("中文", "zh"), ("日本語", "ja"),
         ("Español", "es"), ("العربية", "ar")]
EMO_MODES = ["Referans sesle aynı", "Duygu referans sesi kullan", "Duygu vektörleriyle kontrol",
             "Duygu açıklama metniyle (deneysel)"]
EMO_NAMES = ["Mutlu", "Kızgın", "Üzgün", "Korkmuş", "İğrenmiş", "Melankolik", "Şaşkın", "Sakin"]
UNTRAINED_NORM = 0.5
EXAMPLES = [
    "Mimir. Odin ve Thor'un kötülüklerine dair kaç tane hikâye anlatmışsındır?",
    "Bunu bana bir kez daha tekrar söyler misin?",
    "Belki yarın sabah tekrar uğrarım.",
    "Evet, bugün hava çok güzel ama başka bir gün gibi değil.",
    "Saat 14:30'da istasyonun önünde buluşalım. Fiyatlar yüzde 20 arttı.",
    "Işıklı ırmağın kıyısında ılık bir rüzgâr esiyordu.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=None, help="index-tts kaynak kodu (varsayılan: kardeş klasör)")
    p.add_argument("--model-dir", default=None, help="IndexTTS-2.5 checkpoints (gpt.pth = stok model)")
    p.add_argument("--checkpoint", default=None, help="Açılışta yüklenecek GPT (varsayılan: bulunan en yeni best.pt)")
    p.add_argument("--runs", action="append", default=[],
                   help="Checkpoint aranacak klasör (tekrarlanabilir; varsayılan ../training-pipeline/work/runs ve ./runs)")
    p.add_argument("--gpu", type=int, default=0, help="GPU numarası (nvidia-smi sırası)")
    p.add_argument("--fp32", action="store_true", help="bf16 yerine fp32 çıkarım")
    p.add_argument("--qwen-emo", action="store_true", help="Duygu açıklama metni modunu aç (QwenEmotion yükler)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #

def describe_checkpoint(path: Path) -> str:
    """Seçim listesi için kısa açıklama (büyük dosyayı okumadan, mmap ile)."""
    import torch

    run = path.parent.parent.name
    try:
        state = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    except Exception:  # noqa: BLE001
        return "{}  [{}]".format(path.name, run)
    if isinstance(state, dict) and "trainable" in state:
        early = state.get("early_stop") or {}
        extra = ""
        if path.name == "best.pt" and early.get("best") is not None:
            extra = ", val {:.4f}".format(float(early["best"]))
        return "{}  [{}, adım {}{}]".format(path.name, run, state.get("step", "?"), extra)
    meta = state.get("itts25ft") if isinstance(state, dict) else None
    step = (meta or {}).get("step")
    return "{}  [{}{}]".format(path.name, run, ", adım {}".format(step) if step else "")


def find_checkpoints(search: List[Path], model_dir: Path) -> List[Tuple[str, str]]:
    found: List[Tuple[float, str, str]] = []
    for root in search:
        if not root.is_dir():
            continue
        for path in list(root.glob("*/checkpoints/*.pt")) + list(root.glob("*/exported/*.pth")):
            found.append((path.stat().st_mtime, describe_checkpoint(path), str(path)))
    found.sort(key=lambda item: (not item[2].endswith("best.pt"), -item[0]))
    choices = [(label, value) for _, label, value in found]
    stock = model_dir / "gpt.pth"
    if stock.is_file():
        choices.append(("Stok IndexTTS-2.5 (fine-tune yok)", str(stock)))
    return choices


# --------------------------------------------------------------------------- #
# app state
# --------------------------------------------------------------------------- #

class App:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        from itts25ft import env

        self.args = args
        repo = env.bootstrap(args.repo)
        self.model_dir = env.find_model_dir(args.model_dir, repo)
        self.cfg = env.load_model_config(self.model_dir)
        self.device = "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"

        from indextts.infer_v2_5 import IndexTTS2

        from itts25ft.lang import resolve
        from itts25ft.textfront import TextFrontend, TextFrontendConfig

        print(">> IndexTTS-2.5 yükleniyor ({})".format(self.device))
        self.tts = IndexTTS2(cfg_path=str(self.model_dir / "config.yaml"), model_dir=str(self.model_dir),
                             use_bf16=not args.fp32, device=self.device, use_cuda_kernel=False,
                             use_accel=False, use_qwen_emo=args.qwen_emo)
        self.slot = resolve("tr", None)
        self.front = TextFrontend(self.model_dir, self.slot, TextFrontendConfig(normalizer="turkish", case="tr_lower"))
        self.lock = threading.Lock()
        self.loaded = "Stok IndexTTS-2.5"
        self.loaded_path: Optional[Path] = None
        self.trained_rows: set = set()
        self.outputs = HERE / "outputs" / "webui"
        self.outputs.mkdir(parents=True, exist_ok=True)

    # -- model ------------------------------------------------------------ #

    def load(self, path_str: str) -> str:
        import torch

        from indextts.utils.checkpoint import load_checkpoint

        from itts25ft.modeling import TrainableSpec, apply_trainable_spec, build_gpt, merge_lora

        path = Path(path_str.strip().strip('"')).expanduser()
        if not path.is_file():
            return "**Dosya yok:** `{}`".format(path)
        started = time.time()
        with self.lock:
            state = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
            if isinstance(state, dict) and "trainable" in state:
                # Eğitim checkpoint'i: stok GPT + aynı eğitim ayarı + eğitilen tensörler, LoRA birleştirilir.
                spec_dict = {k: v for k, v in dict(state.get("spec", {})).items()
                             if k in TrainableSpec.__dataclass_fields__ and k not in ("extra_train", "extra_freeze")}
                base = build_gpt(self.cfg, self.model_dir, device="cpu")
                apply_trainable_spec(base, TrainableSpec(**spec_dict))
                base.load_state_dict(state["trainable"], strict=False)
                merge_lora(base)
                merged = {k: v for k, v in base.state_dict().items() if not k.startswith("inference_model.")}
                self.tts.gpt.load_state_dict(merged, strict=False)
                del base, merged
                kind = "eğitim checkpoint'i, adım {}".format(state.get("step", "?"))
            else:
                load_checkpoint(self.tts.gpt, str(path))
                kind = "gpt.pth"
            del state
            self.tts.gpt.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.loaded, self.loaded_path = path.name, path
            rows_file = path.parent.parent / "text_rows.json"
            self.trained_rows = set(json.loads(rows_file.read_text(encoding="utf-8"))["rows"]) if rows_file.is_file() else set()
        return "**Yüklü:** `{}` ({}) — {:.0f} sn. Fine-tune'da eğitilen Türkçe token satırı: {}".format(
            path, kind, time.time() - started, len(self.trained_rows) or "bilinmiyor")

    # -- text ------------------------------------------------------------- #

    def prepare_text(self, text: str, lang: str) -> Tuple[str, bool]:
        """(modele verilecek metin, upstream normalizasyonu açık mı)."""
        if lang == "tr":
            return self.front.clean(text), False
        return text, True

    def token_preview(self, text: str, lang: str) -> str:
        if not text.strip():
            return ""
        tok = self.front.tokenizer
        encoding = getattr(tok, "encoding", tok)
        if lang == "tr":
            cleaned = self.front.clean(text)
            ids = self.front.encode(cleaned, already_clean=True)
        else:
            cleaned = text
            ids = tok.encode("<|{}|> ".format(lang) + text, allowed_special="all")
        norms = self.tts.gpt.text_embedding.weight.detach().float().norm(dim=1).cpu()
        spans, untrained, finetuned = [], 0, 0
        for token_id in ids:
            try:
                piece = encoding.decode_single_token_bytes(token_id).decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                piece = "?"
            if token_id in self.trained_rows:
                color, title = "#2e7d32", "fine-tune'da eğitildi"
                finetuned += 1
            elif token_id < len(norms) and norms[token_id].item() < UNTRAINED_NORM:
                color, title = "#c62828", "eğitilmemiş satır (norm {:.2f})".format(norms[token_id].item())
                untrained += 1
            else:
                color, title = "#455a64", "taban modelde eğitilmiş"
            spans.append('<span title="{} | id {}" style="display:inline-block;margin:2px;padding:1px 5px;'
                         'border-radius:4px;border:1px solid {};color:{}">{}</span>'.format(
                             html.escape(title), token_id, color, color,
                             html.escape(piece.replace(" ", "␣"))))
        legend = ('<div style="margin-top:6px;font-size:0.9em"><b>{}</b> token &nbsp; '
                  '<span style="color:#2e7d32">■ fine-tune\'da eğitildi ({})</span> &nbsp; '
                  '<span style="color:#455a64">■ taban modelde eğitilmiş</span> &nbsp; '
                  '<span style="color:#c62828">■ eğitilmemiş ({})</span></div>').format(len(ids), finetuned, untrained)
        return ('<div><b>Modele giden metin:</b> <code>{}</code></div><div style="margin-top:6px">{}</div>{}'
                .format(html.escape(cleaned), "".join(spans), legend))

    # -- synthesis -------------------------------------------------------- #

    def generate(self, prompt, text, lang, duration_factor, emo_mode, emo_audio, emo_weight, emo_random,
                 vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, emo_text,
                 do_sample, temperature, top_p, top_k, num_beams, repetition_penalty, length_penalty,
                 max_mel_tokens, max_text_tokens_per_segment, interval_silence, seed, progress=None):
        import gradio as gr
        import torch

        from itts25ft.utils import set_seed

        if not prompt:
            raise gr.Error("Önce bir referans ses yükle.")
        if not text or not text.strip():
            raise gr.Error("Metin boş.")
        mode = EMO_MODES.index(emo_mode) if emo_mode in EMO_MODES else 0
        if mode == 3 and self.tts.qwen_emo is None:
            raise gr.Error("Duygu açıklama metni için arayüzü --qwen-emo ile başlat.")
        vec = None
        if mode == 2:
            vec = self.tts.normalize_emo_vec([vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8], apply_bias=True)
        model_text, normalize = self.prepare_text(text, lang)
        out = self.outputs / "{}_{}.wav".format(time.strftime("%Y%m%d-%H%M%S"), lang)
        kwargs = dict(
            do_sample=bool(do_sample), temperature=float(temperature), top_p=float(top_p),
            top_k=int(top_k) if int(top_k) > 0 else None, num_beams=int(num_beams),
            repetition_penalty=float(repetition_penalty), length_penalty=float(length_penalty),
            max_mel_tokens=int(max_mel_tokens),
        )
        with self.lock:
            if int(seed) >= 0:
                set_seed(int(seed))
            self.tts.gr_progress = progress
            started = time.time()
            self.tts.infer(
                spk_audio_prompt=prompt, text=model_text, output_path=str(out), lang=lang,
                emo_audio_prompt=emo_audio if mode == 1 else None, emo_alpha=float(emo_weight),
                emo_vector=vec, use_emo_text=(mode == 3), emo_text=(emo_text or None) if mode == 3 else None,
                use_random=bool(emo_random), interval_silence=int(interval_silence),
                max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                duration_factor=float(duration_factor), text_normalization=normalize, **kwargs,
            )
            elapsed = time.time() - started
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        import soundfile as sf

        seconds = sf.info(str(out)).duration
        info = ("**Model:** `{}` &nbsp;|&nbsp; **Süre:** {:.1f} sn ses, {:.1f} sn üretim (RTF {:.2f})\n\n"
                "**Modele giden metin:** `{}`").format(self.loaded, seconds, elapsed, elapsed / max(seconds, 1e-6),
                                                    model_text)
        return str(out), info


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

def build_ui(app: App, search: List[Path]):
    import gradio as gr

    choices = find_checkpoints(search, app.model_dir)
    default = app.args.checkpoint or (choices[0][1] if choices else "")

    with gr.Blocks(title="IndexTTS-2.5 TR fine-tune") as demo:
        gr.HTML('<h2 style="margin-bottom:0">IndexTTS-2.5 — Türkçe fine-tune</h2>'
                '<p style="margin-top:4px">Sıfır atışlı ses klonlama; Türkçe için eğitimdeki metin işleme kullanılır. '
                'Diğer diller (en/zh/ja/es/ar) aynı modelle denenebilir.</p>')

        with gr.Row(equal_height=False):
            ckpt = gr.Dropdown(choices=choices, value=default or None, label="Checkpoint",
                               info="best.pt = doğrulama kaybı en iyi nokta; eğitim checkpoint'leri bellekte birleştirilir",
                               allow_custom_value=True, scale=4)
            refresh = gr.Button("Listeyi yenile", scale=1)
            load_btn = gr.Button("Yükle", variant="primary", scale=1)
        load_info = gr.Markdown("**Yüklü:** stok IndexTTS-2.5")

        with gr.Tab("Ses üretimi"):
            gr.Markdown("### Ses referansı")
            prompt = gr.Audio(label="Referans ses (5–15 sn, tek konuşmacı)", type="filepath",
                              sources=["upload", "microphone"])
            gr.Markdown("### Metin")
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    text = gr.Textbox(label="Metin", lines=4, placeholder="Seslendirilecek metni yaz")
                    with gr.Row():
                        lang = gr.Dropdown(choices=LANGS, value="tr", label="Dil")
                        duration_factor = gr.Slider(label="Hız (süre katsayısı)", minimum=0.5, maximum=2.0,
                                                    value=1.0, step=0.01, info="hızlı ← 1.0 → yavaş")
                    gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[text], label="Örnek cümleler")
                with gr.Column(scale=1):
                    gen_btn = gr.Button("Ses üret", variant="primary")
                    output = gr.Audio(label="Sonuç", type="filepath")
                    gen_info = gr.Markdown()

            with gr.Accordion("Metin ve token önizleme", open=False):
                gr.Markdown("Metnin modele hangi token'larla gittiğini gösterir. "
                            "<span style='color:#c62828'>Kırmızı</span> = hiç eğitilmemiş satır (karışma riski); "
                            "<span style='color:#2e7d32'>yeşil</span> = fine-tune'da eğitildi.")
                preview_btn = gr.Button("Önizle")
                preview = gr.HTML()

            with gr.Accordion("Duygu ayarları", open=False):
                emo_mode = gr.Radio(choices=EMO_MODES if app.args.qwen_emo else EMO_MODES[:3],
                                    value=EMO_MODES[0], label="Duygu kontrol yöntemi")
                emo_audio = gr.Audio(label="Duygu referans sesi", type="filepath", sources=["upload", "microphone"])
                with gr.Row():
                    emo_weight = gr.Slider(label="Duygu ağırlığı", minimum=0.0, maximum=1.0, value=0.65, step=0.01)
                    emo_random = gr.Checkbox(label="Duygu rastgele örnekleme", value=False)
                vec_sliders = []
                with gr.Row():
                    for name in EMO_NAMES[:4]:
                        vec_sliders.append(gr.Slider(label=name, minimum=0.0, maximum=1.0, value=0.0, step=0.05))
                with gr.Row():
                    for name in EMO_NAMES[4:]:
                        vec_sliders.append(gr.Slider(label=name, minimum=0.0, maximum=1.0, value=0.0, step=0.05))
                emo_text = gr.Textbox(label="Duygu açıklaması (deneysel; --qwen-emo gerekir)",
                                      visible=app.args.qwen_emo)

            with gr.Accordion("Gelişmiş üretim ayarları", open=False):
                with gr.Row():
                    do_sample = gr.Checkbox(label="do_sample", value=True, info="örnekleme")
                    temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.05)
                    top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                    top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                with gr.Row():
                    num_beams = gr.Slider(label="num_beams", minimum=1, maximum=10, value=3, step=1)
                    repetition_penalty = gr.Number(label="repetition_penalty", value=10.0, minimum=0.1, maximum=20.0)
                    length_penalty = gr.Number(label="length_penalty", value=0.0, minimum=-2.0, maximum=2.0)
                    seed = gr.Number(label="Tohum (-1 = rastgele)", value=-1, precision=0)
                with gr.Row():
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", minimum=50,
                                               maximum=int(app.cfg.gpt.max_mel_tokens), value=1500, step=10,
                                               info="küçük olursa ses kesilir")
                    max_text_tokens = gr.Slider(label="Segment başına en fazla metin token'ı", minimum=20,
                                                maximum=int(app.cfg.gpt.max_text_tokens), value=120, step=2)
                    interval_silence = gr.Slider(label="Segmentler arası sessizlik (ms)", minimum=0, maximum=1000,
                                                 value=200, step=10)

        def do_load(path):
            if not path:
                return "**Checkpoint seç.**"
            return app.load(path)

        def do_refresh():
            fresh = find_checkpoints(search, app.model_dir)
            return gr.update(choices=fresh, value=fresh[0][1] if fresh else None)

        def do_generate(*values, progress=gr.Progress()):
            return app.generate(*values, progress=progress)

        load_btn.click(do_load, inputs=[ckpt], outputs=[load_info])
        refresh.click(do_refresh, outputs=[ckpt])
        preview_btn.click(app.token_preview, inputs=[text, lang], outputs=[preview])
        gen_btn.click(
            do_generate,
            inputs=[prompt, text, lang, duration_factor, emo_mode, emo_audio, emo_weight, emo_random,
                    *vec_sliders, emo_text, do_sample, temperature, top_p, top_k, num_beams,
                    repetition_penalty, length_penalty, max_mel_tokens, max_text_tokens, interval_silence, seed],
            outputs=[output, gen_info],
        )
        if default:
            demo.load(lambda: app.load(default), outputs=[load_info])
    return demo


def main() -> int:
    args = parse_args()
    search = [Path(p) for p in args.runs] or [HERE.parent / "training-pipeline" / "work" / "runs", HERE / "runs"]
    app = App(args)
    demo = build_ui(app, search)
    demo.queue(default_concurrency_limit=1)
    # Çıktılar repo içinde; WebUI başka bir klasörden başlatılsa da gradio bu dosyaları sunabilsin.
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=False,
                allowed_paths=[str(app.outputs)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
