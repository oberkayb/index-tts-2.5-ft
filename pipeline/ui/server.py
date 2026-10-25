"""Pipeline arayüzü: adımları başlat/durdur, logu ve eğitimi izle, pipeline.toml'u düzenle.

    uv run --project ../../index-tts python ui/server.py      (pipeline/ klasöründen)
    -> http://127.0.0.1:7870

Yalnızca bu makineden erişilir (127.0.0.1). Aynı anda tek adım çalışır; adımlar
`python pipeline.py <adım>` ile ayrı süreçte koşar, yani arayüz kapansa da
adım kaldığı yerden devam ettirilebilir.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
PIPELINE_DIR = HERE.parent
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(HERE))

import psutil  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import pipeline as pl  # noqa: E402
from tomledit import set_values  # noqa: E402

STEP_RE = re.compile(r"^step\s+(\d+)\s+\|.*?\|\s*loss\s+([\d.]+)(?:.*?\|\s*lr\s+([\d.e+-]+))?(?:.*?\|\s*([\d.]+)\s*it/s)?(?:.*?eta\s+(\S+))?")
VAL_RE = re.compile(r"^\s*val @\s+(\d+)\s+\|\s*loss\s+([\d.]+)")
STEP_OPTIONS = {
    "setup": [], "manifest": ["force"], "preprocess": ["force", "limit"], "pairs": ["force"],
    "train": ["probe"], "export": ["best", "step", "merged"], "eval": ["step", "limit"],
}


def kill_tree(pid: int) -> None:
    """Süreci ve bütün alt süreçlerini kapat (önce nazikçe, 5 sn sonra zorla)."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = parent.children(recursive=True) + [parent]
    for proc in procs:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=5)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass


def request_train_stop(run_dir: Path) -> None:
    """Eğitim her adımda output_dir/STOP'a bakar; görünce checkpoint kaydedip çıkar."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "STOP").write_text("arayüz\n", encoding="utf-8")


class Job:
    """Arayüzün başlattığı tek pipeline adımı ve çıktısı."""

    def __init__(self, step: str, cmd: List[str], run_dir: Path) -> None:
        self.step = step
        self.cmd = cmd
        self.run_dir = run_dir
        self.lines: collections.deque = collections.deque(maxlen=4000)
        self.first = 0  # deque'den düşen satır sayısı (istemci offset'i için)
        self.started = time.time()
        self.ended: Optional[float] = None
        self.code: Optional[int] = None
        self.stop_requested = 0
        kwargs: Dict[str, object] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        self.proc = subprocess.Popen(
            cmd, cwd=str(PIPELINE_DIR), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", bufsize=1, **kwargs,
        )
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            # \r ile güncellenen ilerleme çubuklarında yalnız son hâli tut
            line = line.rstrip("\n").split("\r")[-1]
            if len(self.lines) == self.lines.maxlen:
                self.first += 1
            self.lines.append(line)
        self.code = self.proc.wait()
        self.ended = time.time()

    @property
    def running(self) -> bool:
        return self.code is None

    def stop(self) -> str:
        """İlk istek: nazik durdurma (eğitim checkpoint kaydeder). İkincisi: süreç ağacını kapat."""
        if not self.running:
            return "bitti"
        self.stop_requested += 1
        if self.stop_requested == 1:
            if self.step == "train" and "--probe" not in self.cmd:
                request_train_stop(self.run_dir)
                return "durduruluyor"
            try:
                if os.name == "nt":
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(self.proc.pid, signal.SIGINT)
                return "durduruluyor"
            except (OSError, ValueError):
                pass
        kill_tree(self.proc.pid)
        return "kapatıldı"

    def info(self) -> Dict[str, object]:
        return {"step": self.step, "running": self.running, "code": self.code, "started": self.started,
                "ended": self.ended, "stopping": self.stop_requested > 0 and self.running,
                "command": subprocess.list2cmdline(self.cmd[2:]), "dry_run": "--dry-run" in self.cmd,
                "external": False}


class State:
    def __init__(self, config: Path) -> None:
        self.config = config
        self.job: Optional[Job] = None
        self.lock = threading.Lock()
        self.external_stops: Dict[int, int] = {}

    def pipeline(self) -> pl.Pipeline:
        return pl.Pipeline(self.config)

    def external(self) -> Optional[Dict[str, object]]:
        """Arayüz dışında (terminal, .bat) başlatılmış ve hâlâ çalışan pipeline adımı."""
        try:
            data = json.loads(self.pipeline().running_file.read_text(encoding="utf-8"))
            pid = int(data["pid"])
        except Exception:
            return None
        if self.job is not None and self.job.proc.pid == pid:
            return None
        try:
            proc = psutil.Process(pid)
            # pid yeniden kullanılmış olabilir: süreç dosyadan sonra başlamış olmamalı
            alive = (proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
                     and proc.create_time() <= float(data["started"]) + 5)
        except psutil.Error:
            alive = False
        if not alive:
            return None
        return {"step": data["step"], "running": True, "code": None, "started": float(data["started"]),
                "ended": None, "stopping": self.external_stops.get(pid, 0) > 0, "command": "",
                "dry_run": False, "external": True, "pid": pid, "run_dir": data.get("run_dir")}

    def current(self) -> Optional[Dict[str, object]]:
        """Gösterilecek adım: arayüzün çalışan işi, yoksa dışarıdaki iş, yoksa arayüzün son biten işi."""
        if self.job is not None and self.job.running:
            return self.job.info()
        ext = self.external()
        if ext:
            return ext
        return self.job.info() if self.job else None


app = FastAPI(title="IndexTTS-2.5 pipeline")
STATE: State


@app.middleware("http")
async def no_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


def gpus() -> List[Dict[str, object]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 6:
            idx, name, used, total, util, temp = parts
            rows.append({"index": int(idx), "name": name.replace("NVIDIA GeForce ", ""),
                         "used": float(used) / 1024, "total": float(total) / 1024,
                         "util": float(util), "temp": float(temp)})
    return rows


@app.get("/api/state")
def api_state():
    try:
        snap = STATE.pipeline().snapshot()
        error = None
    except Exception as exc:  # ayar hatası arayüzde gösterilsin
        snap, error = None, "{}: {}".format(type(exc).__name__, exc)
    return {"snapshot": snap, "error": error, "job": STATE.current(), "gpus": gpus(),
            "options": STEP_OPTIONS}


def log_tail() -> Dict[str, object]:
    """En son yazılan pipeline logunun sonu (dışarıdan başlatılmış adımı izlemek için)."""
    try:
        logs_dir = STATE.pipeline().logs_dir
        newest = max(logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
        with newest.open("rb") as handle:
            handle.seek(max(0, newest.stat().st_size - 256 * 1024))
            text = handle.read().decode("utf-8", errors="replace")
    except Exception:
        return {"lines": [], "next": 0, "source": None, "file": True}
    lines = text.splitlines()[-400:]
    return {"lines": [l.split("\r")[-1] for l in lines], "next": 0, "source": newest.name, "file": True}


@app.get("/api/log")
def api_log(offset: int = 0):
    job = STATE.job
    if job is None or (not job.running and STATE.external()):
        return log_tail()
    start = max(offset, job.first)
    lines = list(job.lines)[start - job.first:]
    return {"lines": lines, "next": job.first + len(job.lines), "source": job.step, "file": False}


@app.post("/api/run")
def api_run(body: Dict[str, object]):
    step = str(body.get("step", ""))
    if step not in STEP_OPTIONS:
        raise HTTPException(400, "Bilinmeyen adım: " + step)
    opts = body.get("options") or {}
    with STATE.lock:
        if STATE.job and STATE.job.running:
            raise HTTPException(409, STATE.job.step + " hâlâ çalışıyor. Önce onu durdur.")
        ext = STATE.external()
        if ext and not opts.get("dry_run"):
            raise HTTPException(409, "{} dışarıdan çalışıyor (pid {}). Önce o bitsin ya da durdur.".format(
                ext["step"], ext["pid"]))
        cmd = [sys.executable, "-X", "utf8", str(PIPELINE_DIR / "pipeline.py"), step,
               "--config", str(STATE.config)]
        for flag in ("force", "best", "merged"):
            if opts.get(flag) and flag in STEP_OPTIONS[step]:
                cmd.append("--" + flag)
        for num in ("probe", "limit", "step"):
            value = int(opts.get(num) or 0)
            if value > 0 and num in STEP_OPTIONS[step]:
                cmd += ["--" + num, str(value)]
        if opts.get("dry_run"):
            cmd.append("--dry-run")
        STATE.job = Job(step, cmd, STATE.pipeline().run_dir)
    return STATE.job.info()


@app.post("/api/stop")
def api_stop():
    if STATE.job and STATE.job.running:
        return {"result": STATE.job.stop()}
    ext = STATE.external()
    if not ext:
        raise HTTPException(400, "Çalışan adım yok.")
    pid = int(ext["pid"])
    count = STATE.external_stops[pid] = STATE.external_stops.get(pid, 0) + 1
    if count == 1 and ext["step"] == "train" and ext.get("run_dir"):
        request_train_stop(Path(str(ext["run_dir"])))
        return {"result": "durduruluyor"}
    kill_tree(pid)
    return {"result": "kapatıldı"}


@app.get("/api/config")
def api_config():
    text = STATE.config.read_text(encoding="utf-8")
    return {"path": str(STATE.config), "values": tomllib.loads(text)}


@app.post("/api/config")
def api_config_save(body: Dict[str, Dict[str, object]]):
    if STATE.job and STATE.job.running:
        raise HTTPException(409, "Bir adım çalışırken ayar değiştirilmez; önce bitsin ya da durdur.")
    old = STATE.config.read_text(encoding="utf-8")
    new = set_values(old, body)
    try:
        tomllib.loads(new)
    except tomllib.TOMLDecodeError as exc:
        raise HTTPException(400, "Geçersiz değer: " + str(exc))
    STATE.config.write_text(new, encoding="utf-8")
    try:
        STATE.pipeline()
    except Exception as exc:
        STATE.config.write_text(old, encoding="utf-8")
        raise HTTPException(400, "Ayar reddedildi, dosya geri alındı: {}".format(exc))
    return {"saved": True}


def train_logs(p: pl.Pipeline) -> List[Path]:
    """Bu run'a ait eğitim logları (probe hariç), eskiden yeniye; kaldığı yerden devamlar dahil."""
    if not p.logs_dir.is_dir():
        return []
    run_dir = str(p.run_dir).lower()
    found = []
    for log in sorted(p.logs_dir.glob("*_train.log")):
        with log.open("r", encoding="utf-8", errors="replace") as handle:
            head = handle.readline().lower()
        if run_dir in head:
            found.append(log)
    return found


@app.get("/api/train")
def api_train():
    try:
        p = STATE.pipeline()
    except Exception:
        return {"train": [], "val": [], "last": None, "best": None, "logs": []}
    steps: Dict[int, float] = {}
    val: Dict[int, float] = {}
    last = None
    logs = train_logs(p)
    for log in logs:
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = STEP_RE.match(line)
            if m:
                steps[int(m.group(1))] = float(m.group(2))
                last = {"step": int(m.group(1)), "loss": float(m.group(2)), "lr": m.group(3),
                        "speed": m.group(4), "eta": m.group(5), "log": log.name}
                continue
            m = VAL_RE.match(line)
            if m:
                val[int(m.group(1))] = float(m.group(2))
    train = sorted(steps.items())
    if len(train) > 1500:  # grafik için seyrelt
        stride = len(train) // 1500 + 1
        train = train[::stride] + [train[-1]]
    vals = sorted(val.items())
    best = min(vals, key=lambda kv: kv[1]) if vals else None
    return {"train": train, "val": vals, "last": last, "best": best, "logs": [l.name for l in logs]}


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=pl.DEFAULT_CONFIG)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7870)
    args = parser.parse_args()

    global STATE
    pl.Pipeline(args.config)  # pipeline.toml yoksa örnekten oluşturur, bozuksa hemen söyler
    STATE = State(Path(args.config).resolve())

    import uvicorn

    print("Pipeline arayüzü: http://{}:{}".format(args.host, args.port))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
