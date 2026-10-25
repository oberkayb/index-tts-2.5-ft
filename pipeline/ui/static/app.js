"use strict";

const $ = (s) => document.querySelector(s);
const fmt = (n) => (n == null ? "–" : Number(n).toLocaleString("tr-TR"));

const STEPS = [
  { id: "setup", name: "Kurulum", about: "Model ağırlıklarını model_dir'e yerleştirir ve ölçüm için Whisper'ı indirir." },
  { id: "manifest", name: "Manifest", about: "Dataset satırlarını süre, hız ve metin kurallarına göre süzer." },
  { id: "preprocess", name: "Kodlama", about: "Klipleri semantik kodlara ve özelliklere çevirir. GPU kullanır; kesilirse kaldığı parçadan devam eder." },
  { id: "pairs", name: "Çiftler", about: "Her hedef klibe aynı konuşmacıdan bir referans klip eşler." },
  { id: "train", name: "Eğitim", about: "LoRA adaptörünü eğitir. Durdurunca son adımı kaydeder, yeniden başlatınca oradan sürer." },
  { id: "export", name: "Adaptör", about: "Checkpoint'ten orijinal modele takılan .safetensors adaptörü çıkarır." },
  { id: "eval", name: "Ölçüm", about: "Test cümlelerini seslendirir, Whisper ile hata oranını ölçer; diğer dilleri de kontrol eder." },
];

const OPT_LABEL = {
  force: { type: "check", label: "Baştan üret" },
  best: { type: "check", label: "En iyi checkpoint" },
  merged: { type: "check", label: "Tam gpt.pth olarak" },
  probe: { type: "num", label: "VRAM ölçümü (adım)" },
  limit: { type: "num", label: "İlk N kayıt" },
  step: { type: "num", label: "Adım no" },
};

// Ayar paneli: pipeline.toml'daki alanlar
const FIELDS = [
  { legend: "Run", note: "Çıktılar work/runs/<ad> altına yazılır. Yeni ad, sıfırdan yeni bir eğitim demek.", items: [
    ["run.name", "text", "Run adı"],
    ["run.lang", "text", "Dil kodu"],
  ]},
  { legend: "LoRA", note: "Eğitilen kısım ayrı bir adaptör olur; orijinal model değişmez.", items: [
    ["lora.mode", ["lora", "partial", "full"], "Eğitim biçimi"],
    ["lora.rank", "int", "Rank"],
    ["lora.alpha", "float", "Alpha"],
    ["lora.learning_rate", "float", "Öğrenme hızı"],
    ["lora.train_heads", "bool", "mel_head'i de eğit"],
  ]},
  { legend: "Eğitim", items: [
    ["train.batch_size", "int", "Batch"],
    ["train.grad_accumulation", "int", "Gradyan biriktirme"],
    ["train.epochs", "int", "Epoch"],
    ["train.save_interval", "int", "Kayıt aralığı (adım)"],
    ["train.keep_checkpoints", "int", "Saklanan checkpoint"],
    ["train.early_stop_patience", "int", "Erken durma sabrı"],
  ]},
  { legend: "GPU", note: "nvidia-smi sırası. Virgülle birden çok GPU: 0, 1", items: [
    ["gpu.preprocess", "ints", "Kodlama"],
    ["gpu.train", "ints", "Eğitim"],
    ["gpu.eval", "int", "Ölçüm"],
    ["train.rank_accum", "ints", "İki GPU payı"],
  ]},
  { legend: "Çıktı", items: [
    ["export.format", ["adapter", "merged"], "Biçim"],
  ]},
  { more: true, legend: "Yollar", note: "Göreli yollar pipeline klasörüne göre çözülür.", items: [
    ["paths.dataset_jsonl", "path", "Dataset jsonl"],
    ["paths.dataset_root", "path", "Dataset kökü"],
    ["paths.upstream_repo", "path", "IndexTTS reposu"],
    ["paths.model_dir", "path", "Model klasörü"],
    ["paths.work_dir", "path", "Çalışma klasörü"],
    ["paths.seed_model_dir", "path", "Model kaynağı (boş: Hugging Face)"],
  ]},
  { more: true, legend: "Veri", items: [
    ["manifest.max_duration", "float", "En uzun klip (sn)"],
    ["preprocess.min_seconds", "float", "En kısa klip (sn)"],
    ["preprocess.val_speakers", "int", "Doğrulama konuşmacısı"],
    ["pairs.pairs_per_target", "int", "Hedef başına çift"],
  ]},
  { more: true, legend: "Ölçüm", items: [
    ["eval.num_speakers", "int", "Konuşmacı"],
    ["eval.whisper_model", "text", "Whisper modeli"],
    ["eval.temperature", "float", "Temperature"],
    ["eval.top_p", "float", "top_p"],
    ["eval.cross_lingual", "bool", "Diğer dilleri de ölç"],
  ]},
];

let state = null;
let selected = null;
let config = null;
let logNext = 0;
let logJobStarted = null;
let train = null;

// ------------------------------------------------------------------ tema

const MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
const SUN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';

function isDark() {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
}

function renderTheme() {
  const dark = isDark();
  const btn = $("#theme");
  btn.innerHTML = (dark ? SUN : MOON) + (dark ? "Açık tema" : "Karanlık tema");
  btn.setAttribute("aria-label", dark ? "Açık temaya geç" : "Karanlık temaya geç");
}

$("#theme").addEventListener("click", () => {
  const next = isDark() ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (e) { /* tarayıcı izin vermezse yalnız bu oturum */ }
  renderTheme();
  drawChart(null);
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { renderTheme(); drawChart(null); });
renderTheme();

// ------------------------------------------------------------------ api

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

// ------------------------------------------------------------------ steps

function stepStatus(id, snap, job) {
  if (job && job.step === id) {
    if (job.running) return "run";
    if (job.code !== 0) return "fail";
  }
  if (!snap) return "";
  const p = snap.preprocess;
  switch (id) {
    case "setup": return snap.setup.model_files ? "done" : "";
    case "manifest": return snap.manifest.records ? "done" : "";
    case "preprocess": return p.train ? "done" : (p.features || p.chunks_done ? "partial" : "");
    case "pairs": return snap.pairs.train ? "done" : "";
    case "train": return snap.train.checkpoints.length || snap.train.best ? "done" : "";
    case "export": return snap.export.adapters.length || snap.export.merged.length ? "done" : "";
    case "eval": return Object.keys(snap.eval).length ? "done" : "";
  }
  return "";
}

function stepValue(id, snap) {
  if (!snap) return "";
  const p = snap.preprocess;
  switch (id) {
    case "setup": return snap.setup.model_files ? "model hazır" : "model yok";
    case "manifest": return snap.manifest.records ? fmt(snap.manifest.records) + " kayıt" : "";
    case "preprocess":
      if (p.train) return fmt(p.train) + " eğitim, " + fmt(p.val) + " doğrulama";
      if (p.chunks) return p.chunks_done + " / " + p.chunks + " parça";
      return p.features ? fmt(p.features) + " özellik" : "";
    case "pairs": return snap.pairs.train ? fmt(snap.pairs.train) + " çift" : "";
    case "train": {
      if (train && train.last) return "adım " + fmt(train.last.step);
      const c = snap.train.checkpoints;
      return c.length ? "adım " + fmt(c[c.length - 1].replace("step", "")) : (snap.train.best ? "best.pt" : "");
    }
    case "export": {
      const a = snap.export.adapters;
      return a.length ? a[a.length - 1].replace(".safetensors", "") : (snap.export.merged.slice(-1)[0] || "");
    }
    case "eval": {
      const names = Object.keys(snap.eval);
      if (!names.length) return "";
      const last = snap.eval[names[names.length - 1]];
      const tr = last[snap.lang];
      return tr ? tr.metric + " %" + (tr.error_rate * 100).toFixed(1) : names[names.length - 1];
    }
  }
  return "";
}

function renderChain() {
  const snap = state && state.snapshot;
  const job = state && state.job;
  const ol = $("#chain");
  if (!ol.children.length) {
    for (const s of STEPS) {
      const li = document.createElement("li");
      li.className = "stage";
      li.dataset.id = s.id;
      li.innerHTML = `<button type="button" aria-pressed="false"><span class="jack"><span class="led"></span></span>
        <span class="stage-name">${s.name}</span><span class="stage-val"></span></button>`;
      li.querySelector("button").addEventListener("click", () => select(s.id));
      ol.appendChild(li);
    }
  }
  for (const li of ol.children) {
    const id = li.dataset.id;
    const st = stepStatus(id, snap, job);
    li.className = "stage " + st + (id === selected ? " sel" : "");
    li.querySelector("button").setAttribute("aria-pressed", id === selected);
    li.querySelector(".stage-val").textContent = stepValue(id, snap);
    const label = { run: "çalışıyor", done: "tamam", partial: "yarım", fail: "hata" }[st] || "bekliyor";
    li.querySelector("button").setAttribute("aria-label", STEPS.find((s) => s.id === id).name + ": " + label);
  }
}

function nextStep() {
  const snap = state && state.snapshot;
  for (const s of STEPS) if (stepStatus(s.id, snap, null) !== "done") return s.id;
  return "eval";
}

function select(id) {
  selected = id;
  renderChain();
  renderDrawer(true);
}

function renderDrawer(rebuild) {
  const d = $("#drawer");
  const step = STEPS.find((s) => s.id === selected);
  const job = state && state.job;
  const busy = job && job.running;
  const mine = busy && job.step === selected;
  if (rebuild || !d.dataset.step || d.dataset.step !== selected) {
    d.dataset.step = selected;
    const opts = (state ? state.options[selected] : []) || [];
    d.innerHTML = `<div class="drawer-text"><h3>${step.name}</h3><p>${step.about}</p></div>
      <div class="drawer-opts">${opts.map((o) => OPT_LABEL[o].type === "check"
        ? `<label><input type="checkbox" data-opt="${o}"> ${OPT_LABEL[o].label}</label>`
        : `<label>${OPT_LABEL[o].label} <input type="number" min="0" step="1" data-opt="${o}" placeholder="–"></label>`).join("")}
        <label><input type="checkbox" data-opt="dry_run"> Yalnız komutu göster</label></div>
      <div class="drawer-act"><button type="button" class="btn primary" id="run-btn"></button>
        <button type="button" class="btn stop" id="stop-btn" hidden>Durdur</button></div>`;
    $("#run-btn").addEventListener("click", runStep);
    $("#stop-btn").addEventListener("click", stopStep);
  }
  const runBtn = $("#run-btn");
  const stopBtn = $("#stop-btn");
  runBtn.textContent = step.name + " adımını başlat";
  runBtn.disabled = !!busy;
  runBtn.title = busy && !mine ? job.step + " çalışıyor; aynı anda tek adım koşar." : "";
  stopBtn.hidden = !mine;
  if (mine) {
    const hard = job.stopping || (job.external && job.step !== "train");
    stopBtn.textContent = job.stopping ? "Zorla kapat" : hard ? "Kapat" : "Durdur";
    stopBtn.classList.toggle("hard", !!hard);
  }
}

async function runStep() {
  const options = {};
  for (const el of document.querySelectorAll("#drawer [data-opt]")) {
    options[el.dataset.opt] = el.type === "checkbox" ? el.checked : Number(el.value || 0);
  }
  try {
    await api("/api/run", { step: selected, options });
    logNext = 0;
    $("#console").textContent = "";
    await refresh();
  } catch (e) { banner(e.message); }
}

async function stopStep() {
  const job = state.job;
  // Dışarıdaki eğitim dışı adımların nazik durdurması yok: ilk basışta da süreç kapanır.
  const hard = job.stopping || (job.external && job.step !== "train");
  if (hard && !confirm("Süreç ve alt süreçleri hemen kapatılacak; son kayıttan sonraki ilerleme kaybolur. Devam edilsin mi?")) return;
  try { await api("/api/stop", {}); await refresh(); } catch (e) { banner(e.message); }
}

// ------------------------------------------------------------------ header

function renderHeader() {
  const snap = state.snapshot;
  const job = state.job;
  if (snap) {
    $("#run-name").textContent = snap.run;
    document.title = snap.run + " eğitimi";
  }
  let line;
  if (job && job.running) {
    const step = STEPS.find((s) => s.id === job.step).name;
    const mins = Math.floor((Date.now() / 1000 - job.started) / 60);
    const where = job.external ? " (arayüz dışından başlatıldı)" : "";
    line = `<b>${step}</b> ${job.stopping ? "durduruluyor" : "çalışıyor"}${where}, ${mins} dakikadır`;
    if (job.step === "train" && train && train.last && train.last.eta) line += `. Kalan tahmini ${train.last.eta}`;
  } else if (job && job.code != null) {
    const step = STEPS.find((s) => s.id === job.step).name;
    line = job.code === 0 ? (job.dry_run ? `<b>${step}</b> komutu gösterildi, çalıştırılmadı.` : `<b>${step}</b> bitti.`) : `<b>${step}</b> hata verdi (çıkış ${job.code}). Ayrıntı aşağıdaki çıktıda.`;
  } else if (snap && STEPS.every((s) => stepStatus(s.id, snap, null) === "done")) {
    line = `Dil <b>${snap.lang}</b>. Bütün adımlar tamam; yeni bir eğitim için ayarlardan run adını değiştir.`;
  } else if (snap) {
    line = `Dil <b>${snap.lang}</b>. Sıradaki adım: <b>${STEPS.find((s) => s.id === nextStep()).name}</b>`;
  } else {
    line = "Ayar dosyası okunamadı.";
  }
  $("#run-line").innerHTML = line;

  $("#gpus").innerHTML = state.gpus.map((g) => {
    const pct = Math.min(100, (g.used / g.total) * 100);
    return `<div class="gpu ${g.util > 50 ? "hot" : ""}"><div class="gpu-top"><b>GPU ${g.index}</b><span>${g.name}</span></div>
      <div class="meter" role="meter" aria-valuemin="0" aria-valuemax="${g.total.toFixed(1)}" aria-valuenow="${g.used.toFixed(1)}" aria-label="GPU ${g.index} bellek"><i style="width:${pct}%"></i></div>
      <div class="gpu-sub">${g.used.toFixed(1)} / ${g.total.toFixed(0)} GB, yük %${g.util.toFixed(0)}, ${g.temp.toFixed(0)}°C</div></div>`;
  }).join("") || `<div class="gpu"><div class="gpu-sub">nvidia-smi bulunamadı</div></div>`;

  if (state.error) banner("Ayar dosyası geçersiz: " + state.error, true);
  else if (!$("#banner").dataset.sticky) $("#banner").hidden = true;
}

function banner(msg, sticky) {
  const b = $("#banner");
  b.textContent = msg;
  b.hidden = false;
  b.dataset.sticky = sticky ? "1" : "";
  if (!sticky) setTimeout(() => { if (b.textContent === msg) b.hidden = true; }, 8000);
}

// ------------------------------------------------------------------ log

function classify(line) {
  if (/HATA|Traceback|Error|error:/.test(line)) return "c-err";
  if (line.startsWith(">>") || line.startsWith("$ ")) return "c-cmd";
  if (/val @|en iyi doğrulama/.test(line)) return "c-val";
  if (/kaydedildi|hazır|bitti|tamam/i.test(line)) return "c-ok";
  return "";
}

function appendLines(lines, replace) {
  const pre = $("#console");
  const frag = document.createDocumentFragment();
  for (const l of lines) {
    const span = document.createElement("span");
    const c = classify(l);
    if (c) span.className = c;
    span.textContent = l + "\n";
    frag.appendChild(span);
  }
  if (replace) pre.textContent = "";
  pre.appendChild(frag);
  while (pre.childNodes.length > 3000) pre.removeChild(pre.firstChild);
  if ($("#follow").checked) pre.scrollTop = pre.scrollHeight;
}

async function pollLog() {
  const job = state && state.job;
  if (job && job.started !== logJobStarted) { logJobStarted = job.started; logNext = 0; $("#console").textContent = ""; }
  const data = await api("/api/log?offset=" + logNext);
  if (data.file) {
    const live = job && job.running && job.external;
    $("#log-src").textContent = data.source ? (live ? "canlı, dışarıdan: " : "son log: ") + data.source : "henüz log yok";
    const sig = data.source + ":" + data.lines.length + ":" + (data.lines[data.lines.length - 1] || "");
    if (sig !== $("#console").dataset.sig) { $("#console").dataset.sig = sig; appendLines(data.lines, true); }
  } else {
    $("#log-src").textContent = "canlı: " + data.source;
    $("#console").dataset.sig = "";
    if (data.lines.length) appendLines(data.lines, false);
    logNext = data.next;
  }
}

$("#console").addEventListener("scroll", () => {
  const pre = $("#console");
  const atEnd = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 24;
  $("#follow").checked = atEnd;
});

// ------------------------------------------------------------------ chart

const chart = { pts: null, x: null, y: null };

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function drawChart(hover) {
  const cv = $("#chart");
  const has = train && train.train.length > 1;
  $("#chart-empty").hidden = has;
  cv.hidden = !has;
  if (!has) return;
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight || 260;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext("2d");
  g.scale(dpr, dpr);
  const pad = { l: 52, r: 12, t: 12, b: 26 };
  const raw = train.train;
  const val = train.val;
  // EMA
  const ema = [];
  let e = raw[0][1];
  for (const [s, l] of raw) { e = 0.92 * e + 0.08 * l; ema.push([s, e]); }
  const xs = raw.map((p) => p[0]);
  const all = raw.map((p) => p[1]).concat(val.map((p) => p[1]));
  const sorted = all.slice().sort((a, b) => a - b);
  // aykırı ilk adımları kesmek için %98'lik üst sınır
  let lo = sorted[0], hi = sorted[Math.floor(sorted.length * 0.98)];
  const span = (hi - lo) || 1; lo -= span * 0.06; hi += span * 0.06;
  const x0 = xs[0], x1 = Math.max(xs[xs.length - 1], val.length ? val[val.length - 1][0] : 0);
  const X = (s) => pad.l + ((s - x0) / ((x1 - x0) || 1)) * (W - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (Math.min(v, hi) - lo) / (hi - lo)) * (H - pad.t - pad.b);
  chart.x = X; chart.y = Y; chart.raw = raw; chart.ema = ema; chart.pad = pad; chart.W = W;

  const ink = css("--ink"), muted = css("--muted"), rule = css("--rule"), off = css("--off"), green = css("--green");
  g.font = "11px " + css("--mono");
  g.fillStyle = muted;
  g.strokeStyle = rule;
  g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const v = lo + ((hi - lo) * i) / 4;
    const y = Math.round(Y(v)) + 0.5;
    g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke();
    g.textAlign = "right"; g.textBaseline = "middle";
    g.fillText(v.toFixed(2), pad.l - 8, y);
  }
  g.textAlign = "center"; g.textBaseline = "top";
  for (let i = 0; i <= 4; i++) {
    const s = x0 + ((x1 - x0) * i) / 4;
    g.fillText(Math.round(s).toLocaleString("tr-TR"), Math.min(W - pad.r - 20, Math.max(pad.l + 10, X(s))), H - pad.b + 8);
  }
  const line = (pts, color, w) => {
    g.strokeStyle = color; g.lineWidth = w; g.lineJoin = "round"; g.beginPath();
    pts.forEach(([s, v], i) => (i ? g.lineTo(X(s), Y(v)) : g.moveTo(X(s), Y(v))));
    g.stroke();
  };
  line(raw, off, 1);
  line(ema, ink, 2);
  g.fillStyle = green;
  for (const [s, v] of val) { g.beginPath(); g.arc(X(s), Y(v), 3.5, 0, Math.PI * 2); g.fill(); }
  if (train.best) {
    const [s, v] = train.best;
    g.strokeStyle = green; g.lineWidth = 2;
    g.beginPath(); g.arc(X(s), Y(v), 7, 0, Math.PI * 2); g.stroke();
  }
  if (hover != null) {
    const i = nearest(hover);
    const [s, v] = raw[i];
    g.strokeStyle = muted; g.lineWidth = 1; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(X(s), pad.t); g.lineTo(X(s), H - pad.b); g.stroke(); g.setLineDash([]);
    const vv = val.filter((p) => Math.abs(X(p[0]) - X(s)) < 6)[0];
    $("#chart-read").hidden = false;
    $("#chart-read").textContent = `adım ${fmt(s)}  kayıp ${v.toFixed(3)}  yumuşak ${ema[i][1].toFixed(3)}` + (vv ? `  val ${vv[1].toFixed(3)}` : "");
  } else {
    $("#chart-read").hidden = true;
  }
}

function nearest(px) {
  let best = 0, d = Infinity;
  chart.raw.forEach(([s], i) => { const dd = Math.abs(chart.x(s) - px); if (dd < d) { d = dd; best = i; } });
  return best;
}

$("#chart").addEventListener("mousemove", (ev) => drawChart(ev.offsetX));
$("#chart").addEventListener("mouseleave", () => drawChart(null));
window.addEventListener("resize", () => drawChart(null));

function renderTrainStats() {
  const el = $("#train-stats");
  if (!train || !train.last) { el.innerHTML = ""; return; }
  const l = train.last;
  const parts = [`adım <b>${fmt(l.step)}</b>`, `kayıp <b>${l.loss.toFixed(3)}</b>`];
  if (train.best) parts.push(`en iyi val <b>${train.best[1].toFixed(3)}</b> (${fmt(train.best[0])})`);
  if (l.lr) parts.push(`lr <b>${l.lr}</b>`);
  if (l.speed && state.job && state.job.running && state.job.step === "train") parts.push(`<b>${l.speed}</b> it/s`);
  el.innerHTML = parts.map((p) => `<span>${p}</span>`).join("");
}

async function pollTrain() {
  train = await api("/api/train");
  renderTrainStats();
  drawChart(null);
}

// ------------------------------------------------------------------ settings

function get(obj, path) { return path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj); }

function shown(type, v) {
  if (v === undefined || v === null) return "";
  if (type === "ints") return Array.isArray(v) ? v.join(", ") : String(v);
  return String(v);
}

function parse(type, raw) {
  raw = raw.trim();
  if (type === "int") { if (!/^-?\d+$/.test(raw)) throw new Error("tam sayı"); return parseInt(raw, 10); }
  if (type === "float") { const n = Number(raw.replace(",", ".")); if (raw === "" || !isFinite(n)) throw new Error("sayı"); return n; }
  if (type === "ints") {
    if (raw === "") return [];
    const parts = raw.split(/[,\s]+/).filter(Boolean);
    if (!parts.every((p) => /^\d+$/.test(p))) throw new Error("virgülle ayrılmış sayılar");
    const nums = parts.map(Number);
    return nums;
  }
  return raw;
}

function adapterSize(rank) {
  // r32 adaptör 53,8 MB: LoRA rank başına ~0,94 MB + satırlar ve mel_head ~23,7 MB
  return (rank * 0.94 + 23.7).toFixed(0);
}

function buildSettings() {
  const form = $("#settings");
  form.innerHTML = "";
  let more = null;
  for (const group of FIELDS) {
    const fs = document.createElement("fieldset");
    fs.innerHTML = `<legend>${group.legend}</legend>` + (group.note ? `<p class="fs-note">${group.note}</p>` : "") + `<div class="fields"></div>`;
    const box = fs.querySelector(".fields");
    for (const [path, type, label] of group.items) {
      const v = get(config.values, path);
      const f = document.createElement("label");
      f.className = "field" + (type === "path" ? " wide" : "") + (type === "bool" ? " toggle wide" : "");
      if (type === "bool") {
        f.innerHTML = `<input type="checkbox" data-path="${path}" data-type="bool"><span>${label}</span>`;
        f.querySelector("input").checked = !!v;
      } else if (Array.isArray(type)) {
        f.innerHTML = `<span>${label}</span><select data-path="${path}" data-type="choice">${type.map((o) => `<option>${o}</option>`).join("")}</select>`;
        f.querySelector("select").value = v;
      } else {
        const num = type === "int" || type === "float";
        f.innerHTML = `<span>${label}</span><input type="text" ${num ? 'inputmode="decimal"' : ""} data-path="${path}" data-type="${type}" spellcheck="false">`;
        f.querySelector("input").value = shown(type, v);
      }
      if (path === "lora.rank" || path === "lora.alpha") f.insertAdjacentHTML("beforeend", `<small data-hint="${path}"></small>`);
      box.appendChild(f);
    }
    if (group.more) {
      if (!more) {
        more = document.createElement("details");
        more.className = "more";
        more.innerHTML = "<summary>Yollar, veri ve ölçüm</summary>";
        form.appendChild(more);
      }
      more.appendChild(fs);
    } else form.appendChild(fs);
  }
  for (const el of form.querySelectorAll("[data-path]")) {
    el.dataset.orig = el.type === "checkbox" ? String(el.checked) : el.value;
    el.addEventListener("input", onEdit);
    el.addEventListener("change", onEdit);
  }
  $("#cfg-path").textContent = config.path;
  hints();
  onEdit();
}

function hints() {
  const rank = Number($('[data-path="lora.rank"]').value) || 0;
  const alpha = Number(String($('[data-path="lora.alpha"]').value).replace(",", ".")) || 0;
  $('[data-hint="lora.rank"]').textContent = rank ? `Adaptör yaklaşık ${adapterSize(rank)} MB` : "";
  $('[data-hint="lora.alpha"]').textContent = rank && alpha ? `Ölçek ${(alpha / rank).toFixed(2)} (alpha / rank)` : "";
}

function collect() {
  const updates = {};
  let bad = false, n = 0;
  for (const el of document.querySelectorAll("#settings [data-path]")) {
    const now = el.type === "checkbox" ? String(el.checked) : el.value;
    const changed = now !== el.dataset.orig;
    el.classList.toggle("changed", changed);
    el.classList.remove("bad");
    if (!changed) continue;
    const [sec, key] = el.dataset.path.split(".");
    let val;
    try {
      val = el.dataset.type === "bool" ? el.checked : el.dataset.type === "choice" ? el.value : parse(el.dataset.type, el.value);
    } catch (e) { el.classList.add("bad"); el.title = "Beklenen: " + e.message; bad = true; continue; }
    (updates[sec] = updates[sec] || {})[key] = val;
    n++;
  }
  return { updates, bad, n };
}

function onEdit() {
  hints();
  const { bad, n } = collect();
  const busy = state && state.job && state.job.running;
  $("#save").disabled = bad || n === 0 || busy;
  $("#revert").disabled = n === 0;
  const msg = $("#save-msg");
  msg.classList.toggle("err", bad);
  msg.textContent = bad ? "Kırmızı alanı düzelt." : busy && n ? "Adım çalışırken kaydedilemez." : n ? n + " değişiklik kaydedilmedi" : "";
}

async function saveSettings() {
  const { updates, bad } = collect();
  if (bad) return;
  try {
    await api("/api/config", updates);
    config = await api("/api/config");
    buildSettings();
    $("#save-msg").textContent = "Kaydedildi.";
    await refresh();
  } catch (e) {
    $("#save-msg").classList.add("err");
    $("#save-msg").textContent = e.message;
  }
}

$("#save").addEventListener("click", saveSettings);
$("#revert").addEventListener("click", buildSettings);

// ------------------------------------------------------------------ loop

async function refresh() {
  state = await api("/api/state");
  if (!selected) selected = state.job && state.job.running ? state.job.step : nextStep();
  renderHeader();
  renderChain();
  renderDrawer(false);
  renderTrainStats();
}

async function loop() {
  try {
    await refresh();
    await pollLog();
  } catch (e) {
    banner("Sunucuya ulaşılamıyor: " + e.message);
  }
  setTimeout(loop, state && state.job && state.job.running ? 1500 : 4000);
}

async function trainLoop() {
  try { await pollTrain(); renderChain(); } catch (e) { /* sonraki turda */ }
  setTimeout(trainLoop, state && state.job && state.job.running && state.job.step === "train" ? 10000 : 30000);
}

(async function init() {
  try {
    config = await api("/api/config");
    buildSettings();
  } catch (e) { banner("Ayarlar okunamadı: " + e.message, true); }
  await loop();
  renderDrawer(true);
  trainLoop();
})();
