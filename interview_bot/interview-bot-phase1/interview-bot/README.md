# In-house interview bot (Retell AI replacement)

Real-time voice interviewer that runs on a laptop with open-source models.
Built in phases; this repo is currently at **Phase 1: audio loop (echo bot)**.

## Project layout

Organised into layers so each concern lives in one place. Moving a module here
changed no behaviour — the full test suite is the proof (`pytest -q`).

```
interview-bot/
  .env.example  requirements.txt  pytest.ini
  data/                        # the only writable tree besides logs/ (git-ignored). Two kinds of thing:
    candidates/  eval/  sample_plan.json   #   inputs: resumes, JDs, plans, eval case sets
    sessions/  tts_cache/                  #   runtime output: controller state + cached TTS audio
  logs/                        # runtime output only: server.log, replays, eval runs; logs/debug.log when DEBUG_TRACE=true
  docs/                        # project notes (handoff.md); not code, not runtime output
  interview_bot/
    config.py                  # every threshold + model name, from .env
    server.py                  # FastAPI: web client + WebRTC signalling  (python -m interview_bot.server)
    debug.py                   # opt-in tracing toolkit (off by default)

    models/                    # data contracts (pydantic), no behaviour
      schemas.py               #   QuestionPlan, Question, EvalResult, ...

    services/                  # external / model-backed capabilities ("the N services")
      llm.py                   #   Ollama chat client (shared HTTP pool)
      stt.py                   #   speech-to-text (local faster-whisper | groq) + filters
      tts.py                   #   text-to-speech (Kokoro) + line cache
      bot_speaker.py           #   interviewer LLM (asks / rephrases questions)
      evaluator.py             #   evaluator LLM -> EvalResult

    controller/                # live orchestration: wiring, sessions, state machines
      pipeline.py              #   builds + runs the Pipecat pipeline per WebRTC call
      session.py               #   frame-processor sessions (classic + natural)
      controller.py            #   the classic state machine
      natural.py two_step.py decide.py      # the natural-engine turn logic
      turn_taking.py           #   barge-in / backchannel / echo-guard policy
      echo_guard.py echo.py intents.py      # turn helpers
      timers.py speech_clock.py             # timing helpers
      diagnostics.py           #   mic-input level logger

    repository/                # persistence (JSON today, Redis later)
      session_store.py         #   controller state <-> data/sessions/<slug>_state.json

    prep/  planner.py          # before the interview: resume + JD -> QuestionPlan
    web/   index.html          # minimal WebRTC client
  scripts/                     # offline tools: replay, benchmarks, eval, make_plan
  tests/
```

Import layering: `controller/` depends on `services/`, `models/` and
`repository/`; `services/` depend on `models/`; `models/` depend on nothing in
the app. `config.py` stays at the package root so `PROJECT_ROOT` (its
`__file__.parent.parent`) keeps resolving to the repo root, and `server.py`
stays there so the run command is unchanged.

### Debugging a run

Set `DEBUG_TRACE=true` in `.env`. With it on the server additionally writes a
rotating `logs/debug.log` at DEBUG level and emits `[trace]` lines for each
controller action and candidate turn; `interview_bot.debug` also offers
`trace(event, **fields)`, a `timed(label)` context manager, and
`dump(name, obj)` for JSON snapshots under `logs/debug/`. With it off (the
default) none of this runs, so behaviour is identical to before.

## Phase 1: run it

Python 3.11+ (tested on 3.13). From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # optional; defaults work
python -m interview_bot.server
```

First start downloads faster-whisper large-v3-turbo (~1.6 GB, from Hugging Face)
before it prints `Ready on http://127.0.0.1:7860`. The first WebRTC call downloads
Kokoro (~350 MB, from GitHub) once. Silero VAD and Smart Turn v3 ship inside Pipecat.

Open **http://127.0.0.1:7860** in Chrome, click **Start**, allow the mic.
Use headphones for the first test so echo isn't a variable; then try speakers.

### Test checklist

| Test | Do | Expect |
|---|---|---|
| Echo | Say a sentence, stop | "You said: …" |
| Mic settings | Look at the log box | `echoCancellation=true noiseSuppression=true` |
| Barge-in | Say "long test", talk over the reply | Bot stops ~0.5 s after you start |
| Backchannel | During the long reply say only "okay" / "hmm" | Bot keeps talking; server logs `Ignored backchannel transcript` |
| Early jump-in | Pause mid-sentence until the bot starts, then keep talking | Bot stops immediately |
| Latency | Watch the server log after each reply | `[latency]` lines per stage + TOTAL |

### Switching STT to Groq

```
STT_PROVIDER=groq
GROQ_API_KEY=gsk_...
```

### GPU

With an NVIDIA GPU, set `WHISPER_COMPUTE_TYPE=int8_float16` (device `auto` picks CUDA).

## Tests

```bash
pytest -q
```
