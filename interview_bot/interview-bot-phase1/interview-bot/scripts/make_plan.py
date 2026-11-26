"""Build an interview plan from a resume and a JD (text files).

    python scripts/make_plan.py data/candidates/sai_shashank

reads resume.txt and jd.txt in that folder and writes plan.json next to them.
Then set INTERVIEW_PLAN_PATH=<folder>/plan.json in .env and restart the server.
Run it with the interview server stopped: the 9B prep model and the live model
don't both fit on the GPU.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interview_bot.config import PROJECT_ROOT, get_settings  # noqa: E402
from interview_bot.services.llm import OllamaClient  # noqa: E402
from interview_bot.prep.planner import PrepConfig, make_plan  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder with resume.txt and jd.txt (inside data/)")
    ap.add_argument("--questions", type=int, default=None)
    args = ap.parse_args()
    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    resume = (folder / "resume.txt").read_text(encoding="utf-8")
    jd = (folder / "jd.txt").read_text(encoding="utf-8")

    s = get_settings()
    # Bigger context for resume + JD + thinking; unload right after (keep_alive 0)
    # so the live model gets the GPU back.
    llm = OllamaClient(s.OLLAMA_BASE_URL, num_ctx=s.PREP_NUM_CTX, keep_alive="0")
    cfg = PrepConfig(
        model=s.PREP_LLM_MODEL, num_questions=args.questions or s.PREP_NUM_QUESTIONS,
        think=s.PREP_THINK, max_tokens=s.PREP_MAX_TOKENS,
    )
    print(f"Building the plan with {cfg.model} (thinking {'on' if cfg.think else 'off'}); this can take 1-3 minutes...")
    try:
        plan, notes = await make_plan(llm, resume, jd, cfg)
    finally:
        await llm.close()
    out = folder / "plan.json"
    out.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    print(f"\n{plan.candidate} | {plan.role}\n")
    for q in plan.questions:
        print(f"Q{q.id} [{q.topic}] {q.q}\n    about: {q.context}\n    rubric: {'; '.join(q.expected)}\n")
    for n in notes:
        print(f"note: {n}")
    rel = out.relative_to(PROJECT_ROOT).as_posix() if out.is_relative_to(PROJECT_ROOT) else out
    print(f"\nSaved {rel}\nNext: put INTERVIEW_PLAN_PATH={rel} in .env and restart the server.")


if __name__ == "__main__":
    asyncio.run(main())
