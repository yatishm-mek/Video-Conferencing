// Voice Studio: pick a fine-tuned voice, type text and hear it, with waveform players,
// a comparison shelf per voice and a history of generated clips.
const $ = (id) => document.getElementById(id);

const PRESETS = [
  ["Introduce yourself", (n) => `Hi, I'm ${n}! It's really nice to meet you. I hope you're having a lovely day.`],
  ["Morning", () => "Good morning! The sky is clear, the coffee is hot, and today feels like a good day to start something new."],
  ["A short story", () => "Once upon a time, in a small town by the sea, there lived a girl who collected stories from every traveller who passed through.\n\nOne evening, a stranger arrived with a story nobody had ever heard before."],
  ["Explain AI", () => "Artificial intelligence is a way of teaching computers to learn from examples. Instead of following fixed rules, they find patterns in data, and use those patterns to make useful predictions."],
  ["Excited", () => "Wow, I can't believe it actually worked! This is amazing news, and I'm so happy to share it with you."],
  ["Calm", () => "Take a slow, deep breath. There's no rush at all. Everything will be okay, one step at a time."],
  ["Thank you", () => "Thank you so much for listening. Have a wonderful day, and take care!"],
];

const store = {
  get(k, d) { try { const v = localStorage.getItem("rvs." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("rvs." + k, JSON.stringify(v)); } catch { /* private mode */ } },
};

const fmt = (s) => {
  s = Math.max(0, Math.round(s || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}

const PLAY_ICONS = `<svg class="i-play" viewBox="0 0 24 24"><path d="M8 5.5v13l11-6.5z"/></svg>
  <svg class="i-pause" viewBox="0 0 24 24"><path d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg>`;

// ---------------------------------------------------------------- waveforms

let audioCtx = null;
const peakCache = new Map();

async function peaksFor(url) {
  if (peakCache.has(url)) return peakCache.get(url);
  const job = (async () => {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const buf = await (await fetch(url)).arrayBuffer();
    const audio = await audioCtx.decodeAudioData(buf);
    const data = audio.getChannelData(0);
    const n = 400, step = Math.floor(data.length / n) || 1, peaks = [];
    for (let i = 0; i < n; i++) {
      let max = 0;
      for (let j = i * step; j < Math.min((i + 1) * step, data.length); j += 8) max = Math.max(max, Math.abs(data[j]));
      peaks.push(max);
    }
    const top = Math.max(...peaks) || 1;
    return { peaks: peaks.map((p) => p / top), duration: audio.duration };
  })();
  peakCache.set(url, job);
  return job;
}

class Wave {
  constructor(canvas, url) {
    this.canvas = canvas; this.url = url; this.progress = 0; this.peaks = null;
    new ResizeObserver(() => this.draw()).observe(canvas);
    canvas.addEventListener("click", (e) => {
      const r = canvas.getBoundingClientRect();
      player.seek(this, (e.clientX - r.left) / r.width);
    });
  }
  async load() {
    try { const w = await peaksFor(this.url); this.peaks = w.peaks; this.duration = w.duration; this.draw(); }
    catch { /* leave it blank */ }
    return this;
  }
  draw() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight || +c.getAttribute("height");
    if (!w) return;
    c.width = w * dpr; c.height = h * dpr;
    const g = c.getContext("2d");
    g.scale(dpr, dpr);
    const css = getComputedStyle(document.documentElement);
    const base = css.getPropertyValue("--wave").trim(), done = css.getPropertyValue("--wave-done").trim();
    const bars = Math.max(20, Math.floor(w / 4));
    for (let i = 0; i < bars; i++) {
      const p = this.peaks ? this.peaks[Math.floor((i / bars) * this.peaks.length)] : 0.08;
      const bh = Math.max(3, p * (h - 6));
      g.fillStyle = i / bars < this.progress ? done : base;
      g.beginPath();
      g.roundRect(i * 4, (h - bh) / 2, 2.5, bh, 1.5);
      g.fill();
    }
  }
}

// ---------------------------------------------------------------- one player for the page

const player = {
  audio: new Audio(),
  current: null, // { wave, button, time, row }
  rate: 1,
  async play(target) {
    if (this.current && this.current.wave === target.wave) {
      this.audio.paused ? this.audio.play() : this.audio.pause();
      return;
    }
    this.stop();
    this.current = target;
    this.audio.src = target.wave.url;
    this.audio.playbackRate = target.rate ?? 1;
    target.row?.classList.add("active");
    try { await this.audio.play(); } catch { /* blocked: user clicks again */ }
  },
  stop() {
    if (!this.current) return;
    this.audio.pause();
    this.current.button.classList.remove("playing");
    this.current.row?.classList.remove("active");
    this.current.wave.progress = 0; this.current.wave.draw();
    if (this.current.time) this.current.time.textContent = fmt(this.current.wave.duration);
    this.current = null;
  },
  seek(wave, frac) {
    if (!this.current || this.current.wave !== wave) return;
    if (this.audio.duration) this.audio.currentTime = frac * this.audio.duration;
  },
};
player.audio.addEventListener("play", () => player.current?.button.classList.add("playing"));
player.audio.addEventListener("pause", () => player.current?.button.classList.remove("playing"));
player.audio.addEventListener("ended", () => player.stop());
player.audio.addEventListener("timeupdate", () => {
  const c = player.current;
  if (!c || !player.audio.duration) return;
  c.wave.progress = player.audio.currentTime / player.audio.duration;
  c.wave.draw();
  if (c.time) c.time.textContent = `${fmt(player.audio.currentTime)} / ${fmt(player.audio.duration)}`;
});

function playable(button, canvas, url, time, row) {
  const wave = new Wave(canvas, url);
  const target = { wave, button, time, row };
  button.addEventListener("click", () => { audioCtx?.resume(); player.play(target); });
  wave.load().then(() => { if (time && player.current?.wave !== wave) time.textContent = fmt(wave.duration); });
  return target;
}

// A fresh copy of an element, without the old click handlers.
function fresh(id) {
  const el = $(id), copy = el.cloneNode(true);
  copy.classList.remove("playing");
  el.replaceWith(copy);
  return copy;
}

// A button that plays a URL without a waveform (the voice cards' real-voice buttons).
function miniPlayer(button, url) {
  const canvas = document.createElement("canvas");
  canvas.className = "wave"; canvas.hidden = true;
  button.after(canvas);
  return playable(button, canvas, url, null, null);
}

// ---------------------------------------------------------------- voices

let voices = [];
let voice = null; // the selected voice
let currentOnGpu = "";

function setAccent(color) {
  const root = document.documentElement.style;
  root.setProperty("--accent", color);
  root.setProperty("--wave-done", color);
  root.setProperty("--accent-soft", `color-mix(in srgb, ${color} 13%, var(--card))`);
}

function renderVoices() {
  const box = $("voices");
  box.innerHTML = "";
  $("voice-count").textContent = `${voices.length} fine-tuned voice${voices.length === 1 ? "" : "s"}`;
  for (const v of voices) {
    const card = document.createElement("div");
    card.className = "voice";
    card.dataset.id = v.id;
    card.style.setProperty("--vc", v.color);
    card.setAttribute("role", "radio");
    card.tabIndex = 0;
    const score = v.similarity != null ? ` · similarity ${v.similarity.toFixed(3)}` : "";
    card.innerHTML = `
      <div class="voice-badge">${esc(v.name[0])}</div>
      <div class="voice-text">
        <div class="voice-name">${esc(v.name)}<span class="on-gpu" title="Loaded on the GPU">●</span></div>
        <div class="voice-about">${esc(v.about)}</div>
        <div class="voice-pass">pass ${v.epoch}${score}</div>
      </div>
      <button class="mini" title="Hear ${esc(v.name)}'s real voice" aria-label="Hear ${esc(v.name)}'s real voice">${PLAY_ICONS}</button>`;
    card.addEventListener("click", (e) => { if (!e.target.closest(".mini")) selectVoice(v.id); });
    card.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectVoice(v.id); } });
    box.appendChild(card);
    miniPlayer(card.querySelector(".mini"), v.real);
  }
}

function selectVoice(id) {
  const v = voices.find((x) => x.id === id) || voices[0];
  if (!v) return;
  const changed = voice?.id !== v.id;
  const oldName = voice?.name;
  voice = v;
  store.set("voice", v.id);
  setAccent(v.color);
  document.querySelectorAll(".voice").forEach((c) => {
    c.classList.toggle("on", c.dataset.id === v.id);
    c.setAttribute("aria-checked", c.dataset.id === v.id);
  });
  $("brand-badge").textContent = v.name[0];
  $("hero-title").textContent = `Hear ${v.name}`;
  $("hero-about").textContent = v.id === "rithika"
    ? "A one-minute talk about AI, spoken by the fine-tuned voice with natural flow."
    : `Eight everyday sentences, spoken by ${v.name}'s fine-tuned voice (pass ${v.epoch}).`;
  $("studio-title").textContent = `Make ${v.name} say anything`;
  $("compare-hint").textContent = `${v.name}: the real voice and the training passes`;
  $("model-line").textContent = `Qwen3-TTS 1.7B · LoRA · ${v.run} pass ${v.epoch}`;
  // Swap the name in the text box if it still holds the introduction.
  if (changed && oldName && textEl.value.includes(`I'm ${oldName}`)) {
    textEl.value = textEl.value.replaceAll(`I'm ${oldName}`, `I'm ${v.name}`);
  }
  if (changed) {
    player.stop();
    document.querySelector(".hero").hidden = !v.sample;
    if (v.sample) {
      const hero = playable(fresh("hero-play"), fresh("hero-wave"), v.sample, $("hero-time"), null);
      $("hero-play").setAttribute("aria-label", `Play a sample of ${v.name}'s voice`);
      hero.wave.draw();
    }
    renderShowcase();
  }
  updateCount();
  refreshMarkers();
}

function refreshMarkers() {
  document.querySelectorAll(".voice").forEach((c) => c.classList.toggle("loaded", c.dataset.id === currentOnGpu));
}

// ---------------------------------------------------------------- status

let ready = false;
async function refreshStatus() {
  try {
    const s = await api("/api/status");
    const el = $("status");
    el.className = "status " + (s.state === "ready" ? (s.busy ? "busy" : "ready") : s.state === "error" ? "error" : "");
    const loaded = voices.find((v) => v.id === s.current)?.name || s.current;
    $("status-text").textContent =
      s.state === "loading" ? "Loading the voices…" :
      s.state === "error" ? "Voices failed to load" :
      s.busy ? "Speaking…" : `Ready · ${loaded} loaded`;
    if (s.state === "error") $("error").textContent = s.error;
    currentOnGpu = s.current;
    refreshMarkers();
    ready = s.state === "ready";
    updateSpeakButton();
  } catch {
    $("status").className = "status error";
    $("status-text").textContent = "Server not running";
    ready = false; updateSpeakButton();
  }
}

// ---------------------------------------------------------------- studio

let speaking = false;
const textEl = $("text");

function updateSpeakButton() {
  $("speak").disabled = !ready || speaking || !voice || !textEl.value.trim();
  $("speak").querySelector("span").textContent =
    speaking ? "Speaking…" : !ready ? "Loading the voices…" : `Speak as ${voice?.name || "…"}`;
}

function updateCount() {
  const words = textEl.value.trim() ? textEl.value.trim().split(/\s+/).length : 0;
  $("count").textContent = `${words} word${words === 1 ? "" : "s"} · ${textEl.value.length}/2000`;
  const secs = words / 2.5;
  const switching = voice && currentOnGpu && voice.id !== currentOnGpu ? 4 : 0;
  $("estimate").textContent = words ? `≈ ${fmt(secs)} of speech · about ${fmt(secs * 2.4 + 4 + switching)} to make` : "";
  store.set("text", textEl.value);
  updateSpeakButton();
}

function selectedStyle() { return $("style").querySelector(".on").dataset.v; }

async function speak() {
  if ($("speak").disabled) return;
  speaking = true; updateSpeakButton();
  $("error").textContent = "";
  $("progress").hidden = false;
  $("bar").style.width = "4%";
  const started = Date.now();
  const v = voice;
  try {
    let job = await api("/api/speak", {
      method: "POST",
      body: JSON.stringify({ text: textEl.value, voice: v.id, style: selectedStyle(), flow: $("flow").checked }),
    });
    while (["queued", "switching", "speaking"].includes(job.state)) {
      await new Promise((r) => setTimeout(r, 700));
      job = await api(`/api/jobs/${job.id}`);
      const frac = job.total ? (job.done + 0.5) / job.total : 0.05;
      $("bar").style.width = `${Math.min(96, Math.max(4, frac * 100))}%`;
      const t = fmt((Date.now() - started) / 1000);
      $("progress-text").textContent =
        job.state === "switching" ? `Switching to ${v.name}'s voice · ${t}` :
        job.total ? `${v.name} is speaking part ${Math.min(job.done + 1, job.total)} of ${job.total} · ${t}` :
        `Getting ready · ${t}`;
    }
    if (job.state === "error") throw new Error(job.error);
    $("bar").style.width = "100%";
    $("progress-text").textContent = `Done: ${job.duration}s of speech in ${job.seconds_to_make}s`;
    showResult(job, v);
    loadHistory();
  } catch (e) {
    $("error").textContent = "Couldn't speak that: " + e.message;
  } finally {
    speaking = false; updateSpeakButton();
    refreshStatus();
    setTimeout(() => { if (!speaking) $("progress").hidden = true; }, 2500);
  }
}

let resultTarget = null;
function showResult(job, v) {
  const url = `/history/${job.file}.wav`;
  $("result").hidden = false;
  resultTarget = playable(fresh("res-play"), fresh("res-wave"), url, $("res-time"), null);
  resultTarget.rate = player.rate;
  $("res-download").href = url;
  $("res-download").setAttribute("download", `${v.id}-${job.file}.wav`);
  $("res-info").textContent = `${v.name} · ${job.style[0].toUpperCase() + job.style.slice(1)} style · natural flow ${job.flow ? "on" : "off"} · saved to output\\voice_studio`;
  player.play(resultTarget);
}

// ---------------------------------------------------------------- compare shelf

function renderShowcase() {
  const box = $("showcase");
  box.innerHTML = "";
  for (const it of voice.showcase) {
    const row = document.createElement("div");
    row.className = "item" + (it.real ? " real" : "");
    const pick = it.pick ? '<span class="badge pick">chosen</span>' : "";
    const sim = it.similarity != null ? `<span class="badge">similarity ${it.similarity.toFixed(3)}</span>` : "";
    const errs = it.word_errors != null ? `<span class="badge">${it.word_errors} word errors /100</span>` : "";
    const pauses = it.pauses_per_minute != null ? `<span class="badge">${it.pauses_per_minute.toFixed(1)} pauses/min</span>` : "";
    row.innerHTML = `
      <button class="round" aria-label="Play ${esc(it.title)}">${PLAY_ICONS}</button>
      <div>
        <div class="item-title">${esc(it.title)} ${pick}</div>
        <div class="item-about">${esc(it.about)} · <span class="t">${fmt(it.seconds)}</span></div>
        <div class="badges">${sim}${errs}${pauses}</div>
        <canvas class="wave" height="34"></canvas>
      </div>`;
    box.appendChild(row);
    playable(row.querySelector(".round"), row.querySelector("canvas"), it.url, row.querySelector(".t"), row);
  }
  const others = Object.values(voice.other_people || {});
  $("compare-foot").textContent = voice.id === "rithika"
    ? "Similarity: how close to the real voice (1.000 = identical; another person scores about 0.94). Pauses: per minute inside sentences — the real recordings were read aloud from a script."
    : `Similarity: how close to ${voice.name}'s real voice (1.000 = identical; ${voice.name}'s own real clips score ${voice.real_vs_real?.toFixed(3) ?? "—"}`
      + (others.length ? `, other people ${Math.min(...others).toFixed(3)}–${Math.max(...others).toFixed(3)}` : "")
      + "). Word errors: words Whisper heard wrong, per 100. Trust your ears for the final call.";
}

// ---------------------------------------------------------------- history

async function loadHistory() {
  const items = await api("/api/history").catch(() => []);
  const box = $("history");
  $("history-count").textContent = items.length ? `${items.length} saved in output\\voice_studio` : "";
  if (!items.length) {
    box.innerHTML = '<div class="empty">Clips you make appear here, saved on this PC.</div>';
    return;
  }
  box.innerHTML = "";
  for (const it of items) {
    const color = voices.find((v) => v.id === it.voice)?.color || "var(--accent)";
    const row = document.createElement("div");
    row.className = "clip";
    row.innerHTML = `
      <button class="round" aria-label="Play" style="background:${esc(color)}">${PLAY_ICONS}</button>
      <div style="min-width:0">
        <div class="clip-text"></div>
        <div class="clip-meta"><span class="who" style="--vc:${esc(color)}">${esc(it.voice_name)}</span> · <span class="t">${fmt(it.duration)}</span> · ${esc(it.style)} · flow ${it.flow ? "on" : "off"} · ${esc(it.created)}</div>
        <canvas class="wave" height="26" hidden></canvas>
      </div>
      <div class="clip-actions">
        <button class="icon-btn reuse" title="Put this text back in the editor (and pick this voice)"><svg viewBox="0 0 24 24"><path d="M4 12a8 8 0 0 1 13.7-5.6L20 4v6h-6l2.3-2.3A6 6 0 1 0 18 12h2a8 8 0 0 1-16 0z"/></svg></button>
        <a class="icon-btn" title="Download" href="${it.url}" download="${esc(it.voice)}-${esc(it.name)}.wav"><svg viewBox="0 0 24 24"><path d="M11 4h2v8.2l3.3-3.3 1.4 1.4L12 16l-5.7-5.7 1.4-1.4 3.3 3.3zM5 18h14v2H5z"/></svg></a>
        <button class="icon-btn del" title="Delete"><svg viewBox="0 0 24 24"><path d="M9 3h6l1 2h4v2H4V5h4zM6 9h12l-1 12H7z"/></svg></button>
      </div>`;
    row.querySelector(".clip-text").textContent = it.text;
    row.querySelector(".clip-text").title = it.text;
    box.appendChild(row);
    const canvas = row.querySelector("canvas");
    const target = playable(row.querySelector(".round"), canvas, it.url, row.querySelector(".t"), row);
    row.querySelector(".round").addEventListener("click", () => { canvas.hidden = false; target.wave.draw(); });
    row.querySelector(".reuse").addEventListener("click", () => {
      if (voices.some((v) => v.id === it.voice)) selectVoice(it.voice);
      textEl.value = it.text; updateCount(); textEl.focus(); window.scrollTo({ top: 0, behavior: "smooth" });
    });
    row.querySelector(".del").addEventListener("click", async () => {
      if (!confirm("Delete this clip from output\\voice_studio?")) return;
      if (player.current?.wave === target.wave) player.stop();
      await api(`/api/history/${it.name}`, { method: "DELETE" }).catch(() => {});
      loadHistory();
    });
  }
}

// ---------------------------------------------------------------- wiring

for (const [label, text] of PRESETS) {
  const b = document.createElement("button");
  b.className = "chip"; b.textContent = label;
  b.addEventListener("click", () => { textEl.value = text(voice?.name || "your voice"); updateCount(); textEl.focus(); });
  $("chips").appendChild(b);
}
for (const b of $("style").querySelectorAll("button")) {
  b.addEventListener("click", () => {
    $("style").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    store.set("style", b.dataset.v);
  });
}
for (const b of $("speed").querySelectorAll("button")) {
  b.addEventListener("click", () => {
    $("speed").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    player.rate = +b.dataset.r;
    if (resultTarget) resultTarget.rate = player.rate;
    if (player.current === resultTarget) player.audio.playbackRate = player.rate;
  });
}
$("flow").addEventListener("change", () => store.set("flow", $("flow").checked));
$("speak").addEventListener("click", speak);
textEl.addEventListener("input", updateCount);
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); speak(); }
  if (e.key === " " && document.activeElement === document.body && player.current) {
    e.preventDefault(); player.audio.paused ? player.audio.play() : player.audio.pause();
  }
});

$("flow").checked = store.get("flow", true);
const savedStyle = store.get("style", "balanced");
$("style").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x.dataset.v === savedStyle));

(async () => {
  voices = await api("/api/voices").catch(() => []);
  const s = await api("/api/status").catch(() => null);
  renderVoices();
  const first = voices.find((v) => v.id === store.get("voice", "")) || voices.find((v) => v.id === s?.default) || voices[0];
  textEl.value = store.get("text", `Hi, I'm ${first?.name || "here"}! Type anything here, and I'll say it in my own voice.`);
  if (first) selectVoice(first.id);
  await refreshStatus();
  updateCount();
  loadHistory();
  setInterval(refreshStatus, 3000);
})();
