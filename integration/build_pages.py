r"""Build the launcher's candidate list and publish the generated question plans.

The interview bot's planner (scripts/make_plan.py) writes each candidate's questions
to data/candidates/<slug>/plan.json. This script makes those usable by the frontend:

  1. launcher/candidates.js   -> window.EZY_CANDIDATES = [{slug, name, role}, ...]
     (loaded by index.html via <script src> so it works even from file://)
  2. launcher/plans/<slug>.json              (for local testing)
     Jitsi/docker-jitsi-meet/custom-pages/plans/<slug>.json   (to deploy to Jitsi)

Run after generating/updating any candidate plan:
    python integration/build_pages.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CANDIDATES = REPO / "interview_bot" / "interview-bot-phase1" / "interview-bot" / "data" / "candidates"
LAUNCHER = REPO / "launcher"
CUSTOM_PAGES = REPO / "Jitsi" / "docker-jitsi-meet" / "custom-pages"


def main():
    rows = []
    out_local = LAUNCHER / "plans"
    out_deploy = CUSTOM_PAGES / "plans"
    out_local.mkdir(parents=True, exist_ok=True)
    out_deploy.mkdir(parents=True, exist_ok=True)

    for folder in sorted(p for p in CANDIDATES.iterdir() if p.is_dir()):
        plan = folder / "plan.json"
        if not plan.is_file():
            continue
        try:
            data = json.loads(plan.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"skip {folder.name}: {e}")
            continue
        slug = folder.name
        rows.append({"slug": slug, "name": data.get("candidate", slug), "role": data.get("role", "")})
        shutil.copyfile(plan, out_local / f"{slug}.json")
        shutil.copyfile(plan, out_deploy / f"{slug}.json")
        print(f"published {slug}: {data.get('candidate','')} ({data.get('role','')})")

    js = "window.EZY_CANDIDATES = " + json.dumps(rows, indent=2) + ";\n"
    (LAUNCHER / "candidates.js").write_text(js, encoding="utf-8")
    print(f"\nwrote launcher/candidates.js ({len(rows)} candidates)")
    print("plans copied to launcher/plans/ and Jitsi/docker-jitsi-meet/custom-pages/plans/")


if __name__ == "__main__":
    main()
