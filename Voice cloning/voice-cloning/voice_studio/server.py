"""Voice Studio: a local web app for the fine-tuned voices.

Pick a voice (Rithika, Aditi, Sharanya, Suresh: each Qwen3-TTS 1.7B fine-tuned with LoRA on
that person's recordings), type text and hear it, compare with the real voice and the other
training passes, and keep a history of what you made. All voices share one base model on
the GPU and switch in a few seconds (voice_finetune/voice_bank.py). Runs on this PC's GPU;
nothing is sent anywhere. Voices are listed in voice_studio/voices.json.

    "%USERPROFILE%\\.venvs\\qwen3tts\\Scripts\\python.exe" voice_studio\\server.py
    then open http://127.0.0.1:8097
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from voice_finetune.full_speed import run_at_full_speed  # noqa: E402

PORT = int(os.getenv("VOICE_STUDIO_PORT", "8097"))
STATIC = Path(__file__).resolve().parent / "static"
VOICES_FILE = Path(__file__).resolve().parent / "voices.json"
SAMPLES = ROOT / "output" / "voice_samples"
PEOPLE_SAMPLES = SAMPLES / "people"
HISTORY = ROOT / "output" / "voice_studio"
MAX_CHARS = 2000

# Speaking style: sampling of the talker (sounds, timing) and sub-talker (fine detail).
# Qwen's defaults are 0.9 / top_k 50 / 0.9.
STYLES = {
    "steady": {"temperature": 0.6, "top_k": 30, "subtalker_temperature": 0.7},
    "balanced": {"temperature": 0.75, "top_k": 40, "subtalker_temperature": 0.8},
    "expressive": {"temperature": 0.9, "top_k": 50, "subtalker_temperature": 0.9},
}
CHUNK_WORDS = 40            # longer paragraphs are spoken a few sentences at a time
# Natural flow: shortens long pauses (Rithika's voice learned the pauses of reading aloud).
FLOW_MAX_PAUSE, FLOW_KEEP_PAUSE = 0.26, 0.18
GAP_IN_PARAGRAPH = {True: 0.30, False: 0.40}
GAP_BETWEEN_PARAGRAPHS = {True: 0.55, False: 0.70}
TARGET_RMS_DB = -20.0

# Rithika's comparison shelf: (file, title, what it is). Paths are relative to the AI Bot folder.
RITHIKA_SHOWCASE = [
    ("data/voice_dataset/ref.wav", "Real voice", "A real recording from the training sessions"),
    ("output/voice_samples/new_voice_zero_shot.wav", "Before fine-tuning", "Qwen cloning from one 8-second clip"),
    ("output/voice_samples/run2_epoch5.wav", "Pass 5", "Fine-tuned, 5 passes over 30 min of recordings"),
    ("output/voice_samples/run2_epoch6.wav", "Pass 6", "Fine-tuned, 6 passes: the chosen model"),
    ("output/voice_samples/run2_epoch6_smooth.wav", "Pass 6 · natural flow", "Pass 6 with reading pauses shortened"),
]


# ---------------------------------------------------------------- audio helpers

def frame_db(audio: np.ndarray, sr: int, frame_s: float = 0.01) -> np.ndarray:
    n = max(1, int(frame_s * sr))
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    return 20 * np.log10(np.sqrt(np.mean(np.square(frames), axis=1)) + 1e-9)


def quiet_frames(audio: np.ndarray, sr: int) -> np.ndarray:
    db = frame_db(audio, sr)
    return db <= max(np.percentile(db, 5) + 10.0, db.max() - 45)


def trim_edges(audio: np.ndarray, sr: int) -> np.ndarray:
    loud = np.nonzero(~quiet_frames(audio, sr))[0]
    if not len(loud):
        return audio
    n = int(0.01 * sr)
    lo, hi = max(0, loud[0] * n - int(0.05 * sr)), min(len(audio), (loud[-1] + 1) * n + int(0.12 * sr))
    out = audio[lo:hi].copy()
    f = int(0.01 * sr)
    out[:f] *= np.linspace(0, 1, f)
    out[-f:] *= np.linspace(1, 0, f)
    return out


def shorten_pauses(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Every silence inside the speech longer than FLOW_MAX_PAUSE is cut down to
    FLOW_KEEP_PAUSE by removing its middle, with a short crossfade."""
    quiet, n = quiet_frames(audio, sr), int(0.01 * sr)
    loud = np.nonzero(~quiet)[0]
    if len(loud) < 2:
        return audio, 0
    cuts, run = [], None
    for i in range(loud[0], loud[-1] + 1):
        if quiet[i] and run is None:
            run = i
        elif not quiet[i] and run is not None:
            if (i - run) * 0.01 > FLOW_MAX_PAUSE:
                mid_keep = FLOW_KEEP_PAUSE / 2
                cuts.append((int((run * 0.01 + mid_keep) * sr), int((i * 0.01 - mid_keep) * sr)))
            run = None
    if not cuts:
        return audio, 0
    fade = int(0.015 * sr)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    out, pos = audio[: cuts[0][0]], cuts[0][1]
    for a, b in cuts[1:] + [(len(audio), len(audio))]:
        piece = audio[pos:a]
        if len(out) > fade and len(piece) > fade:
            out = np.concatenate([out[:-fade], out[-fade:] * (1 - ramp) + piece[:fade] * ramp, piece[fade:]])
        else:
            out = np.concatenate([out, piece])
        pos = b
    return out, len(cuts)


def level(audio: np.ndarray) -> np.ndarray:
    speech = audio[np.abs(audio) > 1e-3]
    if len(speech):
        audio = audio * 10 ** ((TARGET_RMS_DB - 20 * np.log10(np.sqrt(np.mean(speech ** 2)) + 1e-9)) / 20)
    peak = np.abs(audio).max()
    return audio * (0.97 / peak) if peak > 0.97 else audio


def split_text(text: str) -> list[list[str]]:
    """Paragraphs (blank lines), each as chunks of whole sentences of about CHUNK_WORDS."""
    out = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = " ".join(para.split())
        if not para:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        chunks, current = [], []
        for s in sentences:
            if current and len(" ".join(current + [s]).split()) > CHUNK_WORDS:
                chunks.append(" ".join(current))
                current = []
            current.append(s)
        chunks.append(" ".join(current))
        out.append(chunks)
    return out


def audio_seconds(path: Path) -> float:
    try:
        return round(sf.info(path).duration, 1)
    except RuntimeError:
        return 0.0


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------- the voices

def load_voices() -> tuple[list[dict], str]:
    """The voices in voices.json that are ready: trained, with a pass chosen (in the file,
    or the one voice_finetune/pick_pass.py suggested)."""
    cfg = read_json(VOICES_FILE)
    ready = []
    for v in cfg.get("voices", []):
        v = dict(v)
        scores = read_json(PEOPLE_SAMPLES / v["id"] / "scores.json")
        v["epoch"] = v.get("epoch", scores.get("suggested_pass"))
        if v["epoch"] is None or not (ROOT / v["run"] / f"epoch-{v['epoch']}" / "lora.safetensors").exists():
            continue
        v.setdefault("sample", f"output/voice_samples/people/{v['id']}/pass{v['epoch']}.wav")
        if not (ROOT / v["sample"]).exists():  # pick_pass.py not run yet
            v["sample"] = ""
        v["scores"] = scores
        ready.append(v)
    default = cfg.get("default") if any(v["id"] == cfg.get("default") for v in ready) else (ready[0]["id"] if ready else "")
    return ready, default


def showcase_for(v: dict) -> list[dict]:
    """The comparison shelf for one voice: the real voice, then its fine-tuned passes."""
    items = []
    if v["id"] == "rithika":
        sim, pauses = read_json(SAMPLES / "similarity.json"), read_json(SAMPLES / "pauses.json")
        for rel, title, about in RITHIKA_SHOWCASE:
            real = rel.endswith("ref.wav")
            stem = Path(rel).stem
            items.append({"rel": rel, "title": title, "about": about, "real": real,
                          "pick": title.startswith("Pass 6 · natural"),
                          "similarity": sim.get("real_vs_real") if real else sim.get("samples", {}).get(stem),
                          "pauses_per_minute": pauses.get("training clips (real voice)" if real else stem, {}).get("pauses_per_minute")})
        return items
    s = v["scores"]
    items.append({"rel": v["real"], "title": "Real voice", "about": f"A real recording of {v['name']}", "real": True,
                  "similarity": s.get("real_vs_real")})
    for epoch, r in sorted(s.get("passes", {}).items(), key=lambda kv: int(kv[0])):
        chosen = int(epoch) == v["epoch"]
        items.append({"rel": f"output/voice_samples/people/{v['id']}/pass{epoch}.wav", "title": f"Pass {epoch}",
                      "about": "Fine-tuned, 8 test sentences" + (": the chosen pass" if chosen else ""), "pick": chosen,
                      "similarity": r.get("similarity"), "word_errors": r.get("word_errors_per_100")})
    return items


@dataclass
class Job:
    id: str
    text: str
    voice: str
    style: str
    flow: bool
    state: str = "queued"          # queued, switching, speaking, done, error
    done: int = 0
    total: int = 0
    file: str = ""
    error: str = ""
    started: float = field(default_factory=time.time)
    seconds_to_make: float = 0.0
    duration: float = 0.0


class Studio:
    def __init__(self) -> None:
        self.bank = None
        self.voices, self.default = load_voices()
        self.state, self.error, self.load_seconds = "loading", "", 0.0
        self.current = ""
        self.lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        threading.Thread(target=self._load, daemon=True).start()

    def voice(self, voice_id: str) -> dict | None:
        return next((v for v in self.voices if v["id"] == voice_id), None)

    def refresh(self) -> None:
        """Re-reads voices.json and the pass scores (new voices appear without a restart)."""
        self.voices, self.default = load_voices()

    def _load(self) -> None:
        t0 = time.time()
        try:
            from voice_finetune.voice_bank import VoiceBank

            if not self.voices:
                raise FileNotFoundError("no trained voices found: see voice_studio/voices.json")
            self.bank = VoiceBank()
            with self.lock:
                self._use(self.voice(self.default))
                self.bank.speak("Hello.", **STYLES["balanced"])  # the first generation is slow
            self.state, self.load_seconds = "ready", round(time.time() - t0, 1)
            print(f"ready with {self.current} ({self.load_seconds}s); voices: "
                  + ", ".join(f"{v['name']} (pass {v['epoch']})" for v in self.voices), flush=True)
        except Exception as e:  # shown in the page
            self.state, self.error = "error", str(e)
            print(f"could not load the voices: {e}", flush=True)

    def _use(self, v: dict) -> None:
        self.bank.use(ROOT / v["run"], v["epoch"])
        self.current = v["id"]

    def submit(self, text: str, voice_id: str, style: str, flow: bool) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], text=text, voice=voice_id, style=style, flow=flow)
        self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def _run(self, job: Job) -> None:
        with self.lock:  # one model on one GPU: one job at a time
            try:
                t0 = time.time()
                v = self.voice(job.voice)
                if self.current != v["id"]:
                    job.state = "switching"
                    self._use(v)
                job.state = "speaking"
                paragraphs = split_text(job.text)
                job.total = sum(len(p) for p in paragraphs)
                pieces, sr = [], 24000
                for p_i, chunks in enumerate(paragraphs):
                    for c_i, chunk in enumerate(chunks):
                        audio, sr = self.bank.speak(chunk, **STYLES[job.style])
                        if job.flow:
                            audio, _ = shorten_pauses(audio, sr)
                        pieces.append(trim_edges(audio, sr))
                        last_chunk = c_i == len(chunks) - 1
                        if not (last_chunk and p_i == len(paragraphs) - 1):
                            gap = (GAP_BETWEEN_PARAGRAPHS if last_chunk else GAP_IN_PARAGRAPH)[job.flow]
                            pieces.append(np.zeros(int(gap * sr), np.float32))
                        job.done += 1
                out = level(np.concatenate(pieces))
                HISTORY.mkdir(parents=True, exist_ok=True)
                slug = re.sub(r"[^a-z0-9]+", "-", job.text.lower())[:40].strip("-") or "clip"
                name = f"{time.strftime('%Y%m%d-%H%M%S')}_{slug}"
                sf.write(HISTORY / f"{name}.wav", out, sr)
                job.duration, job.seconds_to_make = round(len(out) / sr, 1), round(time.time() - t0, 1)
                meta = {"name": name, "text": job.text, "voice": v["id"], "voice_name": v["name"], "style": job.style,
                        "flow": job.flow, "duration": job.duration, "seconds_to_make": job.seconds_to_make,
                        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "model": f"{Path(v['run']).name} pass {v['epoch']}"}
                (HISTORY / f"{name}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
                job.file, job.state = name, "done"
            except Exception as e:
                job.state, job.error = "error", str(e)


# ---------------------------------------------------------------- web app

app = FastAPI(title="Voice Studio")
studio: Studio | None = None


class SpeakRequest(BaseModel):
    text: str
    voice: str = ""
    style: str = "balanced"
    flow: bool = True


@app.get("/api/status")
def status() -> dict:
    busy = any(j.state in ("queued", "switching", "speaking") for j in studio.jobs.values())
    return {"state": studio.state, "error": studio.error, "busy": busy, "load_seconds": studio.load_seconds,
            "current": studio.current, "default": studio.default, "styles": list(STYLES), "max_chars": MAX_CHARS}


@app.get("/api/voices")
def voices() -> list[dict]:
    studio.refresh()
    out = []
    for v in studio.voices:
        s = v["scores"].get("passes", {}).get(str(v["epoch"]), {})
        out.append({"id": v["id"], "name": v["name"], "about": v.get("about", ""), "color": v.get("color", "#7c5cff"),
                    "epoch": v["epoch"], "run": Path(v["run"]).name, "real": f"/files/{v['real']}",
                    "sample": f"/files/{v['sample']}" if v["sample"] else "", "similarity": s.get("similarity"),
                    "word_errors": s.get("word_errors_per_100"),
                    "real_vs_real": v["scores"].get("real_vs_real"),
                    "other_people": v["scores"].get("other_people", {}),
                    "showcase": [{**{k: x for k, x in it.items() if k != "rel"}, "url": f"/files/{it['rel']}",
                                  "seconds": audio_seconds(ROOT / it["rel"])}
                                 for it in showcase_for(v) if (ROOT / it["rel"]).exists()]})
    return out


@app.post("/api/speak")
def speak(req: SpeakRequest) -> dict:
    text = req.text.strip()
    voice_id = req.voice or studio.default
    if studio.state != "ready":
        raise HTTPException(503, f"the voices are {studio.state}")
    if not studio.voice(voice_id):
        raise HTTPException(400, f"unknown voice {voice_id}")
    if not text:
        raise HTTPException(400, "type something to say")
    if len(text) > MAX_CHARS:
        raise HTTPException(400, f"keep it under {MAX_CHARS} characters")
    if req.style not in STYLES:
        raise HTTPException(400, "unknown style")
    return asdict(studio.submit(text, voice_id, req.style, req.flow))


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict:
    j = studio.jobs.get(job_id)
    if not j:
        raise HTTPException(404, "no such job")
    return asdict(j)


@app.get("/api/history")
def history() -> list[dict]:
    items = []
    for meta in sorted(HISTORY.glob("*.json"), reverse=True) if HISTORY.exists() else []:
        m = read_json(meta)
        if m and (HISTORY / f"{m['name']}.wav").exists():
            m.setdefault("voice", "rithika")  # clips made before there were several voices
            m.setdefault("voice_name", "Rithika")
            m["url"] = f"/history/{m['name']}.wav"
            items.append(m)
    return items


@app.delete("/api/history/{name}")
def delete_clip(name: str) -> dict:
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}_[a-z0-9-]*", name):
        raise HTTPException(400, "bad name")
    for ext in (".wav", ".json"):
        (HISTORY / f"{name}{ext}").unlink(missing_ok=True)
    return {"deleted": name}


@app.get("/history/{file}")
def history_file(file: str):
    path = HISTORY / file
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}_[a-z0-9-]*\.wav", file) or not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


def allowed_files() -> set[str]:
    out = set()
    for v in studio.voices:
        out |= {v["real"], v["sample"]} - {""} | {it["rel"] for it in showcase_for(v)}
    return out


@app.get("/files/{rel:path}")
def sample_file(rel: str):
    if rel not in allowed_files() or not (ROOT / rel).exists():
        raise HTTPException(404)
    return FileResponse(ROOT / rel, media_type="audio/wav")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    run_at_full_speed()
    studio = Studio()
    print(f"Voice Studio: http://127.0.0.1:{PORT}  (loading the voices...)", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
