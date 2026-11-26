# EZY-Hire integration

## TL;DR — run the local helper on the GPU box

The helper (`integration/orchestrator`) is **not** the interview — Jitsi runs the
interview. The helper only (a) **serves** the recruiter/candidate pages so
`/interview.html` is a real page (not redirected into a Jitsi room) and (b)
**auto-launches the bot** on Start. Everything else (Whisper/TTS/Ollama/bot) runs
locally; Jitsi + Moonshine run on the Linux box (`192.168.7.174:8443`).

```powershell
pip install -r integration/orchestrator/requirements.txt
python integration/build_pages.py        # publish candidate plans + build candidates.js
python -m uvicorn integration.orchestrator.app:app --host 0.0.0.0 --port 8800
```

Open **http://localhost:8800/** → pick candidate + options → **Start**:
- Human → recruiter page opens (embedded Jitsi + transcript + that candidate's
  generated questions); the candidate link is shown to share.
- Bot → the same, **and the bot auto-joins the Jitsi room** (no manual command).

`--host 0.0.0.0` lets the remote candidate open `http://<gpu-box-ip>:8800/interview.html?...`
This path serves our page and embeds Jitsi from `192.168.7.174:8443`, so there is
no "room = interview.html" redirect and nothing needs to be mounted into Jitsi.

---


Wires the launcher page to the three modules so one page starts an interview:
choose **Human / Bot**, the **STT** engine (Moonshine / Whisper), the **LLM**
(Own / OpenAI-stub) and the **bot voice** (Standard Kokoro / 4 cloned voices),
then start. Bot interviews **join the Jitsi room** as a participant (Jibri records
them, jigasi transcribes them); human interviews open the room for the interviewer.

```
 launcher/index.html ──POST /api/launch──▶ Orchestrator (:8800)
                                            │  sets ASR engine, (bot) starts bridge + auto-record
                                            └──▶ returns the Jitsi room URL
   jigasi ──ws──▶ ASR router (:2700) ──▶ Moonshine WS (:2710)   control: POST :2770/set_engine
                                      └─▶ Whisper  WS (:2701)
   Bot interview: python -m interview_bot.jitsi_run  → headless Chromium joins the room
                  bot STT = Whisper(Pipecat) | Moonshine(WS client)
                  bot TTS = Kokoro | cloned voice via TTS microservice (:8099)
```

## Components & ports

| Component | Path | Port | Venv |
|---|---|---|---|
| Orchestrator | `integration/orchestrator/app.py` | 8800 | light (fastapi/httpx) |
| ASR router | `Jitsi/custom-asr/asr_router.py` | 2700 (+control 2770) | custom-asr |
| Moonshine WS | `Jitsi/custom-asr/asr_ws_server.py` (`ASR_PORT=2710`) | 2710 | custom-asr |
| Whisper WS | `Jitsi/custom-asr/whisper_ws_server.py` | 2701 | custom-asr (+faster-whisper) |
| Cloned-voice TTS | `integration/tts_service/app.py` | 8099 | qwen3tts |
| Interview bot (Jitsi) | `interview_bot/.../interview_bot/jitsi_run.py` | — | interview-bot (+playwright) |
| Jitsi | `Jitsi/docker-jitsi-meet` | 8443 | docker |

jigasi must point at the router: in `docker-jitsi-meet/.env` set
`JIGASI_TRANSCRIBER_VOSK_URL=ws://<HOST_IP>:2700` (already 2700) and run Moonshine on **2710**.

## Run order (single box)

```powershell
# 1. ASR engines + router (custom-asr venv)
$env:ASR_PORT = "2710"; .\Jitsi\custom-asr\run-asr.ps1              # Moonshine on 2710
.\Jitsi\custom-asr\run-whisper-asr.ps1                              # Whisper on 2701
.\Jitsi\custom-asr\run-asr-router.ps1                              # router on 2700 (+2770)

# 2. Cloned-voice TTS (qwen3tts venv) — only needed for cloned voices
"$env:USERPROFILE\.venvs\qwen3tts\Scripts\python.exe" integration\tts_service\app.py

# 3. Ollama (for the bot LLM)   →   ollama serve ; ollama run qwen3.5:9b

# 4. Jitsi
cd Jitsi\docker-jitsi-meet ; .\jitsi.ps1 up

# 5. Orchestrator (serves the launcher too)
python -m uvicorn integration.orchestrator.app:app --host 127.0.0.1 --port 8800
```

Open **http://127.0.0.1:8800**, configure, and Start. The launch returns two links:

- **Recruiter** `…/video-interview/<room>?type=recruiter&userId=&profileId=&recruiterId=` —
  embedded Jitsi **plus** a live-transcript panel and the planned "next questions".
- **Candidate** `…/video-interview/<room>?type=candidate&userId=&profileId=` —
  video only (mic/camera via the Jitsi toolbar).

### Host the pages ON the Jitsi server (important)

The interview URLs are `https://<jitsi>/interview.html?room=...&type=...`. That path is on
the Jitsi host, so **`interview.html` must exist there as a real file** — otherwise Jitsi's web
app treats `/interview.html` as a *room name* (you'd see the room become "interview.html").

This is already wired: `docker-jitsi-meet/docker-compose.yml` mounts the pages into the web
container (from `launcher/`, no copies):

```yaml
- ../../launcher/interview.html:/usr/share/jitsi-meet/interview.html:ro
- ../../launcher/index.html:/usr/share/jitsi-meet/launcher.html:ro
- ../../launcher/plan.json:/usr/share/jitsi-meet/plan.json:ro
```

Recreate the web container so the mounts take effect:

```powershell
cd Jitsi\docker-jitsi-meet ; docker compose up -d --force-recreate web
```

Then: recruiter opens `https://<jitsi>/launcher.html` (or the local `launcher/index.html`),
clicks **Start & get links**, their interview opens and the candidate link is shown to share.
Per-room generated questions go in `launcher/plans/<room>.json` (mount `launcher/plans` too to
serve them); otherwise `plan.json` is used.

---

Both pages (`launcher/interview.html`, served at `/interview.html?room=...`) embed Jitsi from
`PUBLIC_URL` via `external_api.js`, so the app host and the Jitsi host can differ. The recruiter
panel reads questions from `/api/questions/{room}` and transcript from `/api/transcript/{room}`
(poll + the Jitsi caption events; an ASR tap can also `POST` finals there). First use needs the
Jitsi self-signed cert accepted once (open `PUBLIC_URL` directly).


- **Human** → the room opens; transcription uses the chosen engine; record manually.
- **Bot** → the orchestrator runs `interview_bot.jitsi_run`, a headless Chromium joins the
  room, recording auto-starts, and jigasi transcribes. Needs Playwright in the interview-bot
  venv: `pip install playwright && playwright install chromium`.

## STT selection — make the Whisper/Moonshine toggle actually switch

To have the toggle switch the transcription engine, run the ASR stack **on the Jitsi box**
(next to jigasi). One command:

```bash
cd Jitsi/custom-asr && ./run-all-asr.sh      # Moonshine :2710, Whisper :2701, router :2700
```

This puts the **router on :2700**, where jigasi already connects (`JIGASI_TRANSCRIBER_VOSK_URL=
ws://192.168.7.174:2700`), so **no jigasi change is needed** — Moonshine just moves to :2710
behind it. Whisper needs `faster-whisper` installed in the custom-asr venv.

The orchestrator then calls `POST http://<jitsi-host>:2770/set_engine {engine}` on Start (it
auto-targets the Jitsi host from `EZY_PUBLIC_URL`, so no env var is needed), and jigasi
transcribes with the chosen engine — no restart. If `run-all-asr.sh` isn't running, the helper
logs `Could not set ASR engine …` and transcription stays on whatever jigasi points at.

The **bot's own listening** honours the same choice: `whisper` → faster-whisper in-process
(Pipecat); `moonshine` → a one-shot WS client to the Moonshine server. See `services/stt_moonshine.py`.

## Env overrides (all optional)

Orchestrator: `EZY_PUBLIC_URL`, `EZY_ORCH_PORT`, `EZY_ASR_ROUTER_CONTROL`, `EZY_BOT_PYTHON`,
`EZY_INTERVIEW_BOT_DIR`, `EZY_QWEN_TTS_URL`, `EZY_BOT_DISPLAY_NAME`.
Router: `ASR_BACKEND_MOONSHINE`, `ASR_BACKEND_WHISPER`, `ASR_DEFAULT_ENGINE`.
Whisper WS: `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`, `WHISPER_ASR_PORT`.
TTS service: `EZY_TTS_PORT`, `EZY_VOICE_CLONING_DIR`, `EZY_TTS_DEVICE`.
Bot bridge: `EZY_PUBLIC_URL`, `EZY_BRIDGE_DIR`; bot config via `.env` (`MOONSHINE_WS_URL`,
`QWEN_TTS_URL`, `TTS_PROVIDER`, `STT_PROVIDER`, …).

## Phase E (bot-in-room) — prototype first

The headless-participant bridge is the newest, riskiest piece. De-risk in order
(see the plan file for detail):
1. **Headed** Chromium joins the real room, publishes a tone as the fake mic (a human
   confirms they hear it) and logs remote-track RMS — proves join/publish/capture.
2. WS loopback echo; measure mouth-to-ear latency; validate the resamplers.
3. Drop `JitsiBridgeTransport` into a full interview; test the `flush` barge-in tail; go headless.

`agent.js` assumes docker-jitsi-meet XMPP defaults (`meet.jitsi` / `muc.meet.jitsi`) and
prefers the deployment's own `config.js` when it loads; adjust if `XMPP_DOMAIN` was changed.

## Status

All code is in place and syntax-clean. End-to-end verification requires the full stack
(Docker + GPU + the three venvs) and must be done on the host — it cannot run in CI without them.
