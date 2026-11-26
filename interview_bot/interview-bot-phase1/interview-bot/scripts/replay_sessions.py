"""Replay real candidate turns from server.log through the natural engine.

    python scripts/replay_sessions.py                    # the last 2 sessions in logs/server.log
    python scripts/replay_sessions.py --sessions 3 --no-score

Text-level regression: what the candidate said in each turn (as transcribed
live) is fed to the controller + the real Ollama turn call. Audio timing,
barge-in and echo are not replayed. Prints every bot reply, how often the
no-hint guard replaced a sentence, the actions taken and the latency, and
saves the report under logs/.
"""

import argparse
import asyncio
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipecat.frames.frames import LLMTextFrame  # noqa: E402

from interview_bot.config import PROJECT_ROOT, get_settings  # noqa: E402
from interview_bot.controller.controller import Say  # noqa: E402
from interview_bot.controller.natural import NaturalController, Score, Turn, TurnRunner, run_score  # noqa: E402
from interview_bot.controller.pipeline import natural_config, resume_text_for  # noqa: E402
from interview_bot.services.stt import is_hallucination  # noqa: E402
from interview_bot.services.llm import OllamaClient  # noqa: E402
from interview_bot.models.schemas import QuestionPlan  # noqa: E402

_START = "GREETING -> READY_CHECK"
_STT = re.compile(r"Transcription: \[(.*)\]\s*$")
_TURN_END = "User turn inference triggered"


def extract_sessions(log_text: str, phrases: list[str]) -> list[dict]:
    sessions, cur, parts = [], None, []
    for line in log_text.splitlines():
        if _START in line:
            cur = {"start": line[11:23], "turns": []}
            sessions.append(cur)
            parts = []
            continue
        if cur is None:
            continue
        m = _STT.search(line)
        if m:
            t = m.group(1).strip()
            if t and not is_hallucination(t, phrases):
                parts.append(t)
        elif _TURN_END in line and parts:
            cur["turns"].append(" ".join(parts))
            parts = []
    return [s for s in sessions if s["turns"]]


async def replay(session: dict, plan: QuestionPlan, llm, ncfg, score: bool, out: list[str]) -> dict:
    c = NaturalController(plan, ncfg, resume_text=resume_text_for(get_settings().INTERVIEW_PLAN_PATH))
    runner = TurnRunner(llm, ncfg)
    stats = {"turns": 0, "sentences": 0, "replaced": [], "actions": Counter(), "first_speech": [], "total": [], "failed": 0}

    def p(s=""):
        out.append(s)
        print(s, flush=True)

    p(f"\n=== session from {session['start']} ({len(session['turns'])} candidate turns)")
    for a in c.start():
        if isinstance(a, Say):
            p(f"BOT (fixed): {a.text}")
    c.bot_finished()
    scores = []
    for text in session["turns"]:
        if c.state.name in ("CLOSING", "DONE"):
            break
        p(f"\nCANDIDATE: {text}")
        acts = c.user_turn(text, silence_secs=1.0)
        if not any(isinstance(a, Turn) for a in acts):
            acts = c.timer("answer_hold")
        turn = next((a for a in acts if isinstance(a, Turn)), None)
        if turn is None:
            rule = " ".join(a.text for a in acts if isinstance(a, Say))
            p(f"BOT [rule]: {rule}" if rule else "  (waiting)")
            if c.state.name != "CLOSING":
                c.bot_finished()
            continue
        spoken = []

        async def push(frame, *a):
            if isinstance(frame, LLMTextFrame):
                spoken.append(frame.text)

        outcome = await runner.run(turn.request, push, c, lambda: None)
        if outcome is None:
            stats["failed"] += 1
            done = c.turn_failed(turn.request.turn_id)
            action = "FAILED"
        else:
            done = c.turn_done(outcome)
            action = outcome.action
            stats["turns"] += 1
            stats["sentences"] += outcome.sentences
            stats["replaced"] += outcome.replaced
            stats["actions"][action] += 1
            stats["first_speech"].append(outcome.latency.get("first_speech", outcome.latency.get("total", 0)))
            stats["total"].append(outcome.latency.get("total", 0))
        code_says = " ".join(a.text for a in done if isinstance(a, Say))
        p(f"BOT [{action}]: {''.join(spoken).strip()}" + (f"  || code says: {code_says}" if code_says else ""))
        if outcome is not None:
            for s, why in outcome.replaced:
                p(f"  guard ({why}): {s!r}")
            p(f"  latency: first speech {outcome.latency.get('first_speech', 0):.2f}s, total {outcome.latency.get('total', 0):.2f}s")
        scores += [a.request for a in done if isinstance(a, Score)]
        c.bot_finished()
    if score:
        for req in scores:
            r = await run_score(llm, ncfg, req)
            c.score_done(req.question_id, r)
            if r:
                p(f"SCORE q{req.question_id}: {r['verdict']} {r['score']}/10 flags={r['flags']} notes={r['notes']!r}")
    for f in c.flags:
        p(f"FLAG q{f['question_id']}: {f['flag']} {f['detail']}")
    return stats


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(PROJECT_ROOT / "logs" / "server.log"))
    ap.add_argument("--sessions", type=int, default=2, help="replay the last N sessions")
    ap.add_argument("--no-score", action="store_true", help="skip the background scoring calls")
    args = ap.parse_args()

    s = get_settings()
    plan = QuestionPlan.load(s.INTERVIEW_PLAN_PATH)
    ncfg = natural_config(s)
    sessions = extract_sessions(Path(args.log).read_text(encoding="utf-8", errors="replace"), s.hallucination_phrases)
    sessions = sessions[-args.sessions:]
    if not sessions:
        print("No sessions with candidate turns found in the log.")
        return
    llm = OllamaClient(s.OLLAMA_BASE_URL, num_ctx=s.OLLAMA_NUM_CTX, keep_alive=s.OLLAMA_KEEP_ALIVE)
    out: list[str] = [f"Replay of {len(sessions)} session(s) with {s.LIVE_LLM_MODEL}, plan {s.INTERVIEW_PLAN_PATH}"]
    print(out[0])
    await llm.warmup(s.LIVE_LLM_MODEL)
    allstats = []
    try:
        for sess in sessions:
            allstats.append(await replay(sess, plan, llm, ncfg, not args.no_score, out))
    finally:
        await llm.close()

    turns = sum(x["turns"] for x in allstats)
    sentences = sum(x["sentences"] for x in allstats)
    replaced = [r for x in allstats for r in x["replaced"]]
    actions = sum((x["actions"] for x in allstats), Counter())
    fs = [v for x in allstats for v in x["first_speech"]]
    tot = [v for x in allstats for v in x["total"]]
    why = Counter(r[1].split(" ")[0] for r in replaced)
    summary = [
        "\n=== summary",
        f"turns: {turns} (failed/timed out: {sum(x['failed'] for x in allstats)})",
        f"hint guard: replaced {len(replaced)} of {sentences} sentences ({100 * len(replaced) / max(1, sentences):.1f}%), "
        f"in {sum(1 for x in allstats for _ in x['replaced'])} cases; reasons: {dict(why)}",
        f"actions: {dict(actions)}",
        f"LLM first speech: p50 {pct(fs, 0.5):.2f}s, p90 {pct(fs, 0.9):.2f}s; whole turn: p50 {pct(tot, 0.5):.2f}s, "
        f"p90 {pct(tot, 0.9):.2f}s (add ~0.3 s STT + Smart Turn + TTS for end-to-end)",
    ]
    for line in summary:
        print(line)
    out += summary
    path = PROJECT_ROOT / "logs" / f"replay_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"\nSaved {path.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    asyncio.run(main())
