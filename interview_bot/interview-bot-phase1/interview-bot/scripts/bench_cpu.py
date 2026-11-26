"""CPU-only latency benchmark: how fast is each stage without a GPU?

    python scripts/bench_cpu.py                       # everything, default models
    python scripts/bench_cpu.py --skip-whisper --llm qwen3.5:4b qwen3.5:2b llama3.2:3b
    python scripts/bench_cpu.py --whisper small.en base.en --skip-llm

Forces CPU everywhere: Ollama with num_gpu=0, faster-whisper device=cpu int8,
Kokoro (already CPU). Uses the real natural-engine system prompt and real turn
blocks from the Sai plan, so the LLM numbers match what the live bot sends.

The most important number is "new prompt tokens" per turn. The system prompt
(~1,500 tokens) should be cached after the first turn, so later turns should
only process ~200-400 new tokens. If every turn shows ~1,700, the cache is not
working and CPU latency explodes.

Stages run one after another here; in the live bot they share the CPU, so the
real numbers will be somewhat worse. Report is saved to logs/bench_cpu_*.txt.
At the end the CPU copy of each model is unloaded, so the server loads normally.
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from interview_bot.config import PROJECT_ROOT, get_settings  # noqa: E402

OUT: list[str] = []


def p(s=""):
    OUT.append(s)
    print(s, flush=True)


# ---------------------------------------------------------------- Ollama

def ollama(base, body, stream=True):
    req = urllib.request.Request(f"{base}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def installed_models(base):
    with urllib.request.urlopen(f"{base}/api/tags", timeout=10) as r:
        return {m["name"] for m in json.load(r).get("models", [])}


def turn_requests(plan_path):
    """System prompt + 5 real turn blocks, built by the natural controller."""
    from interview_bot.controller.natural import NaturalConfig, NaturalController, Turn, TurnOutcome
    from interview_bot.models.schemas import QuestionPlan

    c = NaturalController(QuestionPlan.load(plan_path), NaturalConfig())
    c.start()
    c.bot_finished()
    said = [
        "Yes, I am ready.",
        "I'm Sai. Recently I built a RAG chatbot for internal documents using BM25 and dense embeddings with a reranker.",
        "The hardest part was chunking, because long tables were split badly, so I moved to section based chunks with overlap.",
        "Sorry, can you repeat the question?",
        "I evaluated it with a small set of 200 questions and measured recall at 5 and answer faithfulness.",
    ]
    msgs = []
    actions = ["NEXT_QUESTION", "NEXT_QUESTION", "FOLLOW_UP", "REPEAT_QUESTION", "WAIT"]
    for text, action in zip(said, actions):
        acts = c.user_turn(text, silence_secs=1.0) + c.timer("answer_hold")
        t = next((a for a in acts if isinstance(a, Turn)), None)
        if t is None:
            c.bot_finished()
            continue
        msgs.append(t.request.messages)
        # Pretend the model chose a sensible action so the next block moves on.
        if action not in t.request.allowed:
            action = "WAIT"
        spoken = {"FOLLOW_UP": "How did you pick the chunk size?", "WAIT": "Take your time."}.get(action, "Okay.")
        c.turn_done(TurnOutcome(t.request.turn_id, action, spoken=spoken), spoken_by_runner=True)
        c.bot_finished()
    return msgs


def bench_llm(base, model, msgs, threads, score_between):
    opts = {"num_gpu": 0, "num_ctx": 4096, "temperature": 0.5, "num_predict": 60}
    if threads:
        opts["num_thread"] = threads
    p(f"\n--- LLM {model} on CPU (num_gpu=0{', threads=' + str(threads) if threads else ''})")
    t0 = time.monotonic()
    body = {"model": model, "messages": [{"role": "user", "content": "Hi"}], "stream": False, "think": False,
            "options": dict(opts, num_predict=1), "keep_alive": "10m"}
    with ollama(base, body) as r:
        json.load(r)
    p(f"load on CPU: {time.monotonic() - t0:.1f}s")

    rows = []
    for i, m in enumerate(msgs):
        if score_between and i == 2:
            # Same system prompt, different user block, like the background scorer.
            sb = {"model": model, "messages": [m[0], {"role": "user", "content": "TASK: SCORE\nReturn {\"score\": 5} only."}],
                  "stream": False, "think": False, "options": dict(opts, num_predict=10), "keep_alive": "10m"}
            ts = time.monotonic()
            with ollama(base, sb) as r:
                d = json.load(r)
            p(f"  (score call in between: {time.monotonic() - ts:.2f}s, new prompt tokens {d.get('prompt_eval_count')})")
        body = {"model": model, "messages": m, "stream": True, "think": False, "options": opts, "keep_alive": "10m"}
        t0 = time.monotonic()
        first_tok = first_speech = None
        text = ""
        with ollama(base, body) as r:
            for raw in r:
                line = raw.strip()
                if not line:
                    continue
                ch = json.loads(line)
                piece = ch.get("message", {}).get("content", "")
                if piece and first_tok is None:
                    first_tok = time.monotonic() - t0
                text += piece
                if first_speech is None and "\n" in text:
                    reply = text.split("\n", 1)[1]
                    if any(x in reply for x in ",.?!") or len(reply.split()) >= 6:
                        first_speech = time.monotonic() - t0
                if ch.get("done"):
                    total = time.monotonic() - t0
                    pe_n, pe_s = ch.get("prompt_eval_count", 0), ch.get("prompt_eval_duration", 0) / 1e9
                    ev_n, ev_s = ch.get("eval_count", 0), ch.get("eval_duration", 0) / 1e9
                    break
        first_speech = first_speech or total
        rows.append((first_speech, pe_n))
        p(f"  turn {i + 1}: first speech {first_speech:5.2f}s | first token {first_tok or 0:5.2f}s | total {total:5.2f}s | "
          f"new prompt tokens {pe_n:5d} @ {pe_n / max(pe_s, 1e-6):6.0f} tok/s | output {ev_n / max(ev_s, 1e-6):5.1f} tok/s")
        p(f"           {text.strip().splitlines()[0][:40] if text.strip() else ''} / {' '.join(text.split(chr(10))[1:2])[:70]!r}")
    later = [r[0] for r in rows[1:]] or [rows[0][0]]
    cached = all(r[1] < 900 for r in rows[1:]) if len(rows) > 1 else None
    p(f"  => first turn {rows[0][0]:.2f}s, later turns median {statistics.median(later):.2f}s; "
      f"prompt cache {'WORKING' if cached else 'NOT working (whole prompt re-read every turn)'}")
    with ollama(base, {"model": model, "messages": [], "keep_alive": 0, "stream": False}) as r:
        r.read()  # unload the CPU copy
    return statistics.median(later), cached


# ---------------------------------------------------------------- Kokoro

def load_kokoro(s):
    from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR
    import kokoro_onnx

    return kokoro_onnx.Kokoro(str(KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"), str(KOKORO_CACHE_DIR / "voices-v1.0.bin"))


def bench_kokoro(k, s):
    p("\n--- Kokoro TTS (CPU)")
    k.create("Hello.", voice=s.KOKORO_VOICE, speed=s.KOKORO_SPEED, lang="en-us")
    res = {}
    for name, text in [
        ("short first piece (5 words)", "Okay, so you used BM25."),
        ("one sentence (12 words)", "Okay, so you used BM25. How did you choose the chunk size?"),
        ("long sentence (25 words)", "We're going a bit off track, and we have about eighteen of our twenty minutes left. Could you briefly introduce yourself and your recent work?"),
    ]:
        ts = []
        for _ in range(3):
            t0 = time.monotonic()
            audio, sr = k.create(text, voice=s.KOKORO_VOICE, speed=s.KOKORO_SPEED, lang="en-us")
            ts.append(time.monotonic() - t0)
        med = statistics.median(ts)
        res[name] = med
        p(f"  {name:30s} {med:5.2f}s for {len(audio) / sr:4.1f}s of audio")
    return res["short first piece (5 words)"]


# ---------------------------------------------------------------- Whisper

def test_audio(k, s):
    """Make test answers with Kokoro, resampled to 16 kHz for Whisper."""
    clips = {}
    for name, text in [
        ("short reply", "Yes, I'm ready."),
        ("8 s answer", "I built a RAG chatbot with BM25 and dense embeddings, then added a reranker and tuned the chunk size to around 500 tokens."),
    ]:
        a, sr = k.create(text, voice="am_adam", speed=1.0, lang="en-us")
        n = int(len(a) * 16000 / sr)
        clips[name] = (np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32), text)
    return clips


def bench_whisper(models, clips, threads):
    from faster_whisper import WhisperModel

    best = None
    for name in models:
        p(f"\n--- Whisper {name} (CPU int8, beam 1, {threads} threads)")
        try:
            t0 = time.monotonic()
            m = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=threads)
            p(f"  load {time.monotonic() - t0:.1f}s")
        except Exception as e:  # noqa: BLE001
            p(f"  could not load: {e}")
            continue
        m.transcribe(clips["short reply"][0], beam_size=1, language="en")
        worst = 0.0
        for clip, (audio, truth) in clips.items():
            ts, out = [], ""
            for _ in range(2):
                t0 = time.monotonic()
                segs, _ = m.transcribe(audio, beam_size=1, language="en", vad_filter=False)
                out = " ".join(x.text.strip() for x in segs)
                ts.append(time.monotonic() - t0)
            med = statistics.median(ts)
            worst = max(worst, med)
            p(f"  {clip:12s} ({len(audio) / 16000:3.1f}s audio) {med:5.2f}s -> {out!r}")
        if best is None or worst < best[1]:
            best = (name, worst)
        del m
    return best


# ---------------------------------------------------------------- main

def main():
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", nargs="*", default=["qwen3.5:4b", "qwen3.5:2b", "qwen3:4b", "llama3.2:3b"])
    ap.add_argument("--whisper", nargs="*", default=[s.WHISPER_MODEL, "distil-large-v3", "small.en", "base.en"])
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for Ollama/Whisper (0 = auto)")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-whisper", action="store_true")
    ap.add_argument("--no-score-between", action="store_true", help="don't insert a scoring call between turns")
    a = ap.parse_args()

    cores = os.cpu_count() or 4
    wthreads = a.threads or max(1, cores // 2)
    import platform

    p(f"CPU benchmark {time.strftime('%Y-%m-%d %H:%M')} | {platform.processor() or platform.machine()} | {cores} logical cores")
    k = load_kokoro(s)
    tts = bench_kokoro(k, s)

    stt = None
    if not a.skip_whisper:
        stt = bench_whisper(a.whisper, test_audio(k, s), wthreads)

    llm = {}
    if not a.skip_llm:
        base = s.OLLAMA_BASE_URL.rstrip("/")
        have = installed_models(base)
        msgs = turn_requests(s.INTERVIEW_PLAN_PATH)
        p(f"\nLLM test: {len(msgs)} real turns, system prompt ~{len(msgs[0][0]['content']) // 4} tokens")
        for m in a.llm:
            if m not in have and f"{m}:latest" not in have:
                p(f"\n--- LLM {m}: not installed, skipped (run: ollama pull {m})")
                continue
            try:
                llm[m] = bench_llm(base, m, msgs, a.threads, not a.no_score_between)
            except Exception as e:  # noqa: BLE001
                p(f"  failed: {e}")

    p("\n=== estimated end-to-end on this CPU (candidate stops -> bot starts speaking)")
    p("    0.20 s silence wait + ~0.1 s Smart Turn + Whisper (8 s answer) + LLM first speech + Kokoro short piece")
    stt_s = stt[1] if stt else 0.5
    p(f"    Whisper: {stt[0] if stt else 'not measured, assuming'} {stt_s:.2f}s | Kokoro short piece {tts:.2f}s")
    for m, (fs, cached) in llm.items():
        tot = 0.3 + stt_s + fs + tts
        p(f"    {m:14s} {tot:5.2f}s {'OK (<2 s)' if tot < 2 else 'too slow'}{'' if cached else '  [prompt cache not working]'}")
    path = PROJECT_ROOT / "logs" / f"bench_cpu_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    path.write_text("\n".join(OUT), encoding="utf-8")
    p(f"\nSaved {path.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
