# Deploy the interview pages to the Jitsi (Linux) box

These files (`interview.html`, `plan.json`) must be served **as files** by the Jitsi web
server. Otherwise Jitsi treats `https://<jitsi>/interview.html` as a *room* named
"interview.html" (the bug you saw).

They are copies of `launcher/interview.html` and `launcher/plan.json`. Re-run
`sync.ps1` (on Windows) whenever you change those, then redeploy.

## Two-machine setup

- **Windows (GPU):** you open `launcher/index.html` locally (file://). Start builds the
  URLs and opens the Jitsi interview with the room name from the form. The bot / Whisper /
  Qwen-TTS / Ollama all run here, locally.
- **Linux (CPU, 192.168.7.174):** runs Jitsi + Moonshine, and now also **serves these
  two pages**.

## Steps (on the Linux box)

1. Copy this `docker-jitsi-meet` folder (or at least `custom-pages/` and the updated
   `docker-compose.yml`) to the Linux box, keeping `custom-pages/` next to `docker-compose.yml`.

   From Windows, e.g.:
   ```
   scp custom-pages/interview.html custom-pages/plan.json user@192.168.7.174:/path/to/docker-jitsi-meet/custom-pages/
   ```

2. The web service already has the mounts (in `docker-compose.yml`):
   ```yaml
   - ./custom-pages/interview.html:/usr/share/jitsi-meet/interview.html:ro
   - ./custom-pages/plan.json:/usr/share/jitsi-meet/plan.json:ro
   ```

3. Recreate the web container:
   ```
   docker compose up -d --force-recreate web
   ```

4. Verify: `https://192.168.7.174:8443/interview.html?room=test&type=recruiter` should show
   the interview page (not a room called "interview.html").

## Per-room questions (optional)
The interview bot's planner can write `plans/<room>.json`; mount `./custom-pages/plans` the
same way and the recruiter/candidate panels will use the per-room set, else `plan.json`.
