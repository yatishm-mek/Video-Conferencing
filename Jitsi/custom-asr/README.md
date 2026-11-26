# Custom Moonshine ASR for Jitsi (per-speaker, real-time)

Replaces the bundled Vosk (kaldi-en) engine with **your fine-tuned Moonshine
model** (`D:\Realtime-ASR`), so Jitsi's live captions use your model +
contextual biasing + guardrails — still **per speaker** (interviewer vs.
candidate), because Jigasi opens one connection per participant.

## How it fits together

```
Browser (interviewer) ─┐
Browser (candidate)   ─┤─► JVB ─► Jigasi transcriber (one WS per participant)
                                      │  ws://host.docker.internal:2700
                                      ▼
                        asr_ws_server.py  (native Windows, RTX 5050)
                        Vosk protocol ─► resample 16k ─► Silero VAD endpointing
                                       ─► Moonshine (+contextual biasing)
                                       ─► guardrails ─► {"text": ...}
                                      │
                                      ▼
                        per-speaker captions + transcripts (saved by Jigasi)
```

The server speaks Jigasi's Vosk WebSocket protocol
(`{"config":{"sample_rate":N}}` → raw 16-bit PCM → `{"text":...}` / `{"eof":1}`),
so Jigasi talks to it unmodified.

## What's here

| File | Purpose |
|---|---|
| `asr_ws_server.py` | The Vosk-protocol WebSocket server (Silero VAD + endpointing) |
| `asr_engine.py` | Model load + biased transcribe (core copied from `live_asr.py`) |
| `contextual_bias_processor.py` | Copied from `D:\Realtime-ASR\scripts\` (interview-term biasing) |
| `guardrails.py` | Copied — banned-word filtering |
| `banned_words.txt` | Copied blocklist |
| `data/benchmark_vocabulary_tokens.json`, `data/benchmark_trie.json` | **Fixed default** biasing vocab (67-term generic set) — what `run-asr.ps1` always uses |
| `generated_skills/` | Per-candidate sets produced by `make_skills.py`; **not** used by the server unless you point `ASR_VOCAB`/`ASR_TRIE` at them |
| `libs/` | Vendored `websockets` (so the Realtime-ASR venv is untouched) |
| `model/` | Local copy of the fine-tuned checkpoint (inference files; no optimizer state) |
| `run-asr.ps1` | Launcher (uses the Realtime-ASR venv Python) |

**Nothing in `D:\Realtime-ASR` is modified.** The fine-tuned checkpoint is now
copied locally into `custom-asr/model/` (inference files only — `training_state.pt`
omitted), so this folder is self-contained. Override the location with the
`ASR_CHECKPOINT` env var if needed. The only remaining external dependency is the
Python interpreter + packages in `D:\Realtime-ASR\.venv`.

## Run an interview with your model

1. **Start the ASR server** (leave this terminal open; it prints each line it
   transcribes):
   ```powershell
   D:\Jitsi\custom-asr\run-asr.ps1
   ```
   Wait for `[server] listening on ws://0.0.0.0:2700`.

2. **Start Jitsi** (separate terminal):
   ```powershell
   cd D:\Jitsi\docker-jitsi-meet; .\jitsi.ps1 up
   ```

3. Open **https://localhost:8443**, join as Interviewer + Candidate (two tabs/
   devices), allow mic, and click **Subtitles/CC → Start**. Captions now come
   from your Moonshine model, tagged per speaker. Jigasi still saves the
   transcript + per-speaker WAVs under
   `..\docker-jitsi-meet\.jitsi-meet-cfg\storage\transcripts\`.

Order doesn't strictly matter, but the ASR server must be running **before** you
start transcription in a call, or Jigasi can't connect.

## Generate per-candidate skills (dynamic biasing)

For a new candidate, build their biasing vocabulary from the job description +
résumé (the skills-generation half of `D:\Realtime-ASR\dynamic_interview_asr.py`,
copied here as `make_skills.py`):

```powershell
# paste the JD, type END, paste the resume, type END:
python make_skills.py
# or from files:
python make_skills.py --jd jd.json --resume resume.json
```

It extracts skill terms (dropping stopwords, natural languages, filler), tokenizes
them with the model's tokenizer, builds the trie, and writes to the **separate**
`generated_skills/` folder (`candidate_skills.json` + `candidate_trie.json`).

**These are NOT used by `run-asr.ps1`** — it keeps the fixed `data/` benchmark
default. To run the server with a generated set instead, point the env vars at it:

```powershell
$env:ASR_VOCAB="generated_skills\candidate_skills.json"
$env:ASR_TRIE="generated_skills\candidate_trie.json"
.\run-asr.ps1
```

- `make_skills.py` runs on plain `python` (only needs the tokenizer); the server needs the venv Python.
- `--out-name <base>` writes `generated_skills/<base>_skills.json` / `_trie.json`.
- `--keep-dated` also saves a timestamped copy.
- JD/résumé are JSON; skills are read from fields like `skills`, `technologies`,
  `tools`, `requirements`, `qualifications`, `must_have`, etc.

## Tuning (env vars, set before run-asr.ps1)

| Var | Default | Meaning |
|---|---|---|
| `ASR_USE_BIAS` | `1` | `0` disables contextual biasing (plain fine-tuned model) |
| `ASR_SILENCE_MS` | `600` | Pause length (ms) that ends a turn — lower = snappier captions |
| `ASR_VAD_THRESHOLD` | `0.5` | Silero speech-probability gate — raise in a noisy room |
| `ASR_MIN_SPEECH_MS` | `250` | Ignore voice blips shorter than this |
| `ASR_MAX_NEW_TOKENS` | `96` | Cap tokens generated per turn |
| `ASR_CHECKPOINT` | `D:\Realtime-ASR\...\checkpoint-550` | Model directory |
| `ASR_PORT` | `2700` | Server port |

Example (snappier, no biasing):
```powershell
$env:ASR_SILENCE_MS="400"; $env:ASR_USE_BIAS="0"; .\run-asr.ps1
```

## Notes

- **Finals per utterance.** Captions appear when a speaker pauses (Moonshine
  transcribes a whole segment at once). This is accurate and low-overhead; true
  word-by-word partials aren't emitted.
- **Fall back to plain Vosk:** in `..\docker-jitsi-meet\.env` set
  `JIGASI_TRANSCRIBER_VOSK_URL=ws://vosk:2700`, add `-f vosk.yml` back to
  `jitsi.ps1`, and `.\jitsi.ps1 up`.
- **GPU:** the server uses CUDA automatically (RTX 5050). VRAM use is modest
  (~1–2 GB); the big host-RAM Vosk container is gone.
