#!/usr/bin/env python3
"""
Fine-tune edilmiş IndexTTS-2.5'i MCP sunucusu olarak yayınlar (Claude Desktop, Cursor, VS Code, ...).

    mcp_server.bat                                   # stdio (istemci başlatır)
    mcp_server.bat --transport http --port 8765      # http://127.0.0.1:8765/mcp (tek model, çok istemci)

Model sunucu açılınca arka planda yüklenir (~1 dk) ve bellekte kalır; araçlar hazır olana kadar bekler.

Araçlar:
  synthesize        metni seslendir, wav yolunu döndür (istenirse sesi de)
  list_voices       voices/ klasöründeki referans sesler
  list_checkpoints  kullanılabilir GPT ağırlıkları ve yüklü olan
  load_checkpoint   başka bir checkpoint'e geç

stdio'da stdout MCP kanalıdır; mcp>=2 sunucu çalışırken fd 1'i stderr'e çevirir, yani model
logları protokolü bozmaz. Bu yüzden model, sunucu başladıktan sonra (lifespan içinde) yüklenir.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mcp.server.mcpserver import Audio, Context, MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from webui_ft import App, find_checkpoints  # noqa: E402

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
# UnifiedVoice'un duygu vektörü sırası (webui_ft.EMO_NAMES ile aynı)
EMOTIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None,
                   help="Yüklenecek GPT (varsayılan: training-pipeline'da export edilmiş en yeni *best*.pth)")
    p.add_argument("--gpu", type=int, default=0, help="GPU numarası (nvidia-smi sırası; 0 = 5070 Ti)")
    p.add_argument("--voices", action="append", default=[],
                   help="Referans ses klasörü (tekrarlanabilir; varsayılan ./voices)")
    p.add_argument("--default-voice", default=None, help="voice verilmezse kullanılacak ses (ad ya da yol)")
    p.add_argument("--outputs", default=str(HERE / "outputs" / "mcp"), help="Üretilen wav'ların klasörü")
    p.add_argument("--runs", action="append", default=[], help="Checkpoint aranacak klasör (webui_ft ile aynı)")
    p.add_argument("--repo", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--qwen-emo", action="store_true", help="emotion_text desteği (QwenEmotion yükler)")
    p.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


ARGS = parse_args()
SEARCH = [Path(p) for p in ARGS.runs] or [HERE.parent / "training-pipeline" / "work" / "runs", HERE / "runs"]
VOICE_DIRS = [Path(p) for p in ARGS.voices] or [HERE / "voices"]
OUTPUTS = Path(ARGS.outputs)


def log(msg: str) -> None:
    print("[indextts-mcp] " + msg, file=sys.stderr, flush=True)


def default_checkpoint(model_dir: Path) -> str:
    if ARGS.checkpoint:
        return ARGS.checkpoint
    exported = [p for root in SEARCH if root.is_dir() for p in root.glob("*/exported/*best*.pth")]
    if exported:
        return str(max(exported, key=lambda p: p.stat().st_mtime))
    choices = find_checkpoints(SEARCH, model_dir)
    return choices[0][1] if choices else str(model_dir / "gpt.pth")


class Engine:
    """Model bir kez, arka planda yüklenir; GPU işi tek tek (kilitle) yapılır."""

    def __init__(self) -> None:
        self.app: Optional[App] = None
        self.error: Optional[str] = None
        self.ready = threading.Event()
        self.started = False
        self.lock = threading.Lock()

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        threading.Thread(target=self._load, name="indextts-load", daemon=True).start()

    def _load(self) -> None:
        try:
            # Model logları stderr'e (stdio'da stdout MCP kanalı).
            with contextlib.redirect_stdout(sys.stderr):
                started = time.time()
                app = App(SimpleNamespace(repo=ARGS.repo, model_dir=ARGS.model_dir, gpu=ARGS.gpu,
                                          fp32=ARGS.fp32, qwen_emo=ARGS.qwen_emo))
                info = app.load(default_checkpoint(app.model_dir))
                log("hazır ({:.0f} sn): {}".format(time.time() - started, info))
            self.app = app
        except Exception as exc:  # noqa: BLE001
            self.error = "{}: {}".format(type(exc).__name__, exc)
            log("model yüklenemedi: " + self.error)
        finally:
            self.ready.set()

    def wait(self, timeout: float = 600.0) -> App:
        self.start()
        if not self.ready.wait(timeout):
            raise ToolError("Model hâlâ yükleniyor; biraz sonra tekrar dene.")
        if self.app is None:
            raise ToolError("Model yüklenemedi: " + (self.error or "bilinmeyen hata"))
        return self.app


ENGINE = Engine()


@contextlib.asynccontextmanager
async def lifespan(_server):
    ENGINE.start()  # sunucu stdout'u devraldıktan sonra yükle
    yield {}


mcp = MCPServer(
    name="indextts-tr",
    title="IndexTTS-2.5 Türkçe",
    instructions=(
        "Yerel IndexTTS-2.5 (Türkçe fine-tune) ile metni, verilen referans sesin tınısıyla seslendirir. "
        "Önce list_voices ile ses adlarını al; synthesize wav dosyasının yolunu döndürür. "
        "Türkçe için sayıları ve kısaltmaları yazıyla vermek (\"%20\" yerine \"yüzde yirmi\") en doğru sonucu verir. "
        "Diller: tr, en, zh, ja, es, ar."
    ),
    lifespan=lifespan,
)


def voice_files() -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for root in VOICE_DIRS:
        if root.is_dir():
            for path in sorted(root.iterdir()):
                if path.suffix.lower() in AUDIO_EXT:
                    found.setdefault(path.stem, path)
    return found


def resolve_voice(voice: str) -> Path:
    voice = (voice or ARGS.default_voice or "").strip().strip('"')
    if not voice:
        names = ", ".join(voice_files()) or "yok — voices/ klasörüne wav/mp3 koy"
        raise ToolError("voice gerekli (ses adı ya da dosya yolu). Kayıtlı sesler: " + names)
    path = Path(voice).expanduser()
    if path.is_file():
        return path.resolve()
    by_name = {k.lower(): v for k, v in voice_files().items()}
    if voice.lower() in by_name:
        return by_name[voice.lower()]
    raise ToolError("Ses bulunamadı: {}. Kayıtlı sesler: {}".format(voice, ", ".join(voice_files()) or "yok"))


def output_file(output_path: str, lang: str) -> Path:
    if output_path:
        path = Path(output_path).expanduser()
        if not path.is_absolute():
            path = OUTPUTS / path
        if path.suffix.lower() != ".wav":
            path = path.with_suffix(".wav")
    else:
        path = OUTPUTS / "{}_{}.wav".format(time.strftime("%Y%m%d-%H%M%S"), lang)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


@mcp.tool(title="Metni seslendir", structured_output=False)
async def synthesize(
    text: str,
    voice: str = "",
    lang: str = "tr",
    output_path: str = "",
    emotions: Optional[Dict[str, float]] = None,
    emotion_weight: float = 0.65,
    emotion_text: str = "",
    duration_factor: float = 1.0,
    seed: int = -1,
    return_audio: bool = False,
    ctx: Optional[Context] = None,
) -> list:
    """Metni referans sesin tınısıyla seslendirir ve wav dosyasının yolunu döndürür.

    Args:
        text: Seslendirilecek metin (uzun metin otomatik bölünür).
        voice: list_voices'taki ses adı ya da referans ses dosyasının tam yolu (5-15 sn temiz kayıt).
        lang: tr (fine-tune), en, zh, ja, es, ar.
        output_path: İsteğe bağlı çıktı yolu; göreli yol outputs/mcp altına yazılır.
        emotions: İsteğe bağlı duygu vektörü, 0-1 arası: happy, angry, sad, afraid, disgusted,
            melancholic, surprised, calm. Örn. {"happy": 0.6, "calm": 0.2}. Verilmezse referans sesin duygusu.
        emotion_weight: Duygunun etkisi (0-1; varsayılan 0.65).
        emotion_text: Duyguyu metinle tarif et (yalnız sunucu --qwen-emo ile açıldıysa).
        duration_factor: Konuşma süresi katsayısı; 1.0 normal, <1 hızlı, >1 yavaş (0.5-2.0).
        seed: Tekrarlanabilir sonuç için tohum; -1 rastgele.
        return_audio: true ise wav içeriği de yanıtta döner (istemci ses gösterebiliyorsa).
    """
    if not text or not text.strip():
        raise ToolError("text boş.")
    lang = (lang or "tr").strip().lower()
    prompt = resolve_voice(voice)
    vec_values = None
    if emotions:
        unknown = sorted(set(emotions) - set(EMOTIONS))
        if unknown:
            raise ToolError("Bilinmeyen duygu: {}. Geçerli: {}".format(", ".join(unknown), ", ".join(EMOTIONS)))
        vec_values = [max(0.0, min(1.0, float(emotions.get(name, 0.0)))) for name in EMOTIONS]
    if not 0.5 <= float(duration_factor) <= 2.0:
        raise ToolError("duration_factor 0.5 ile 2.0 arasında olmalı.")

    import anyio

    if ctx is not None and not ENGINE.ready.is_set():
        await ctx.report_progress(0.0, 1.0, "model yükleniyor...")
    app = await anyio.to_thread.run_sync(ENGINE.wait)
    if emotion_text and app.tts.qwen_emo is None:
        raise ToolError("emotion_text için sunucuyu --qwen-emo ile başlat.")
    out = output_file(output_path, lang)

    def report(value, desc=None):
        if ctx is not None:
            with contextlib.suppress(Exception):
                anyio.from_thread.run(ctx.report_progress, float(value), 1.0, desc)

    def run() -> dict:
        import soundfile as sf
        import torch

        from itts25ft.utils import set_seed

        model_text, normalize = app.prepare_text(text, lang)
        with ENGINE.lock, app.lock, contextlib.redirect_stdout(sys.stderr):
            if int(seed) >= 0:
                set_seed(int(seed))
            vec = app.tts.normalize_emo_vec(vec_values, apply_bias=True) if vec_values else None
            app.tts.gr_progress = report
            started = time.time()
            try:
                app.tts.infer(
                    spk_audio_prompt=str(prompt), text=model_text, output_path=str(out), lang=lang,
                    emo_alpha=float(emotion_weight), emo_vector=vec,
                    use_emo_text=bool(emotion_text), emo_text=emotion_text or None,
                    duration_factor=float(duration_factor), text_normalization=normalize,
                    do_sample=True, temperature=0.8, top_p=0.8, top_k=30, num_beams=3,
                    repetition_penalty=10.0, length_penalty=0.0, max_mel_tokens=1500,
                    max_text_tokens_per_segment=120,
                )
            finally:
                app.tts.gr_progress = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elapsed = time.time() - started
        seconds = sf.info(str(out)).duration
        return {"path": str(out), "seconds": round(seconds, 2), "generation_seconds": round(elapsed, 2),
                "model": app.loaded, "voice": str(prompt), "lang": lang, "model_text": model_text}

    result = await anyio.to_thread.run_sync(run)
    content: List[object] = [result]
    if return_audio:
        content.append(Audio(path=result["path"]))
    return content


@mcp.tool(title="Referans sesleri listele")
def list_voices() -> dict:
    """synthesize'ın voice parametresine verilebilecek kayıtlı referans sesler (voices/ klasörü)."""
    return {
        "voices": {name: str(path) for name, path in voice_files().items()},
        "voice_dirs": [str(p) for p in VOICE_DIRS],
        "default_voice": ARGS.default_voice,
        "hint": "Yeni ses eklemek için 5-15 sn temiz bir wav/mp3'ü bu klasörlerden birine koy; dosya adı = ses adı.",
    }


@mcp.tool(title="Checkpoint'leri listele")
def list_checkpoints() -> dict:
    """Kullanılabilir GPT ağırlıkları (eğitim checkpoint'leri, export edilmiş .pth, stok model) ve yüklü olan."""
    app = ENGINE.app
    model_dir = app.model_dir if app else Path(ARGS.model_dir or HERE.parent / "index-tts" / "checkpoints")
    return {
        "loaded": str(app.loaded_path) if app and app.loaded_path else None,
        "status": "hazır" if app else ("hata: " + ENGINE.error if ENGINE.error else "yükleniyor"),
        "device": app.device if app else "cuda:{}".format(ARGS.gpu),
        "checkpoints": [{"label": label, "path": path} for label, path in find_checkpoints(SEARCH, model_dir)],
    }


@mcp.tool(title="Checkpoint yükle")
async def load_checkpoint(path: str) -> str:
    """Başka bir GPT ağırlığına geçer (list_checkpoints'teki path). .pth ~10 sn, eğitim .pt ~1 dk sürer."""
    import anyio

    app = await anyio.to_thread.run_sync(ENGINE.wait)

    def run() -> str:
        with ENGINE.lock, contextlib.redirect_stdout(sys.stderr):
            return app.load(path)

    return await anyio.to_thread.run_sync(run)


def main() -> int:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    for root in VOICE_DIRS:
        root.mkdir(parents=True, exist_ok=True)
    log("transport={} gpu={} voices={} outputs={}".format(
        ARGS.transport, ARGS.gpu, ", ".join(map(str, VOICE_DIRS)), OUTPUTS))
    if ARGS.transport == "http":
        ENGINE.start()
        mcp.run("streamable-http", host=ARGS.host, port=ARGS.port)
    else:
        mcp.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
