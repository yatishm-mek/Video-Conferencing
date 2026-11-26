"""Compare small models for the two-step turn: UNDERSTAND accuracy on real
candidate utterances from your sessions, and sample TALK replies.

    python scripts/eval_models.py                    # CPU only (num_gpu=0), default models
    python scripts/eval_models.py --gpu              # quicker check on the GPU
    python scripts/eval_models.py --models qwen3.5:2b gemma4:e2b --skip-talk
    python scripts/eval_models.py --gpu --models gemma4:e4b --cases data/eval/boundary_cases.jsonl
        (can't-do asks, personal problems, role/salary questions; written by hand, not from logs)

Cases: data/eval/understand_cases.jsonl (94 utterances from logs/server.log,
labelled by hand; some allow two labels). Report: logs/eval_models_*.txt.
Stop the server first so the CPU is free.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interview_bot.config import PROJECT_ROOT, get_settings  # noqa: E402
from interview_bot.controller.two_step import (  # noqa: E402
    UNDERSTAND_SCHEMA, Conversation, parse_label, TurnContext, decide_move, label_messages, say_messages, system_prompt,
)  # noqa: E402
from interview_bot.models.schemas import QuestionPlan  # noqa: E402

OUT: list[str] = []
SYS = ""  # the live bot's shared system prompt (set in main)


def p(s=""):
    OUT.append(s)
    print(s, flush=True)


def post(base, path, body, timeout=600):
    req = urllib.request.Request(f"{base}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def capabilities(base, model):
    try:
        return post(base, "/api/show", {"model": model}).get("capabilities", [])
    except Exception:  # noqa: BLE001
        return []


_NO_THINK: set[str] = set()  # models that reject the think flag


def chat(base, model, messages, opts, fmt=None, think_ok=False):
    """think=False always (as in the live bot), unless the model rejects the flag.
    Qwen3.5 thinks by default: without the flag every call silently writes a hidden
    chain of thought first (seen: 2-18 s per call even on the GPU)."""
    body = {"model": model, "messages": messages, "stream": False, "options": opts, "keep_alive": "10m"}
    if fmt:
        body["format"] = fmt
    if model not in _NO_THINK:
        body["think"] = False
    t0 = time.monotonic()
    try:
        d = post(base, "/api/chat", body)
    except urllib.error.HTTPError as e:
        if e.code == 400 and "think" in body:
            _NO_THINK.add(model)
            body.pop("think")
            t0 = time.monotonic()
            d = post(base, "/api/chat", body)
        else:
            raise
    if d.get("message", {}).get("thinking"):
        print(f"  (warning: {model} is still thinking; times include hidden reasoning)")
    global LAST
    LAST = d
    return d.get("message", {}).get("content", ""), time.monotonic() - t0, d.get("prompt_eval_count", 0)


LAST: dict = {}


def split():
    """Ollama's own timing of the last call: reading the prompt vs writing the reply."""
    pe, ev = LAST.get("prompt_eval_duration", 0) / 1e9, LAST.get("eval_duration", 0) / 1e9
    return f"read prompt {pe:.2f}s + write {LAST.get('eval_count', 0)} tokens {ev:.2f}s"


def context(case) -> TurnContext:
    bot = (case.get("bot") or "").lower()
    return TurnContext(
        before_q1=case["qid"] == 0,
        bot_asked_end="end the interview" in bot,
        bot_offered_moment="moment" in bot,
        answered=bool(case.get("earlier")) and len(case["earlier"].split()) >= 8,
        words=len(case["text"].split()),
    )


def greeting(plan):
    first = plan.first_name or "there"
    return (f"Hi {first}, I'm the AI interviewer for the {plan.role or 'this'} role. We'll go through "
            f"{len(plan.questions)} questions, about twenty minutes. Are you ready to begin?")


def conversation(case, plan):
    qtext = plan.questions[case["qid"] - 1].q if case["qid"] else None
    turns = [("bot", qtext or greeting(plan))]
    if case.get("earlier"):
        turns.append(("candidate", case["earlier"]))
    if case.get("bot"):
        turns.append(("bot", case["bot"]))
    turns.append(("candidate", case["text"]))
    return Conversation(turns), qtext


TALK_SAMPLES = [
    # (case id to borrow the conversation from, move, extra format args)
    (38, "bridge", {}),
    (34, "follow_up", {}),
    (23, "answer_question", {"facts": "You are the AI interviewer for this role and have no personal name.", "then": "ask if they are ready to begin."}),
    (26, "rephrase", {}),
    (56, "clarify", {}),
    (61, "decline", {}),
    (81, "confirm_end", {}),
    (90, "off_topic", {"left": 14, "total": 20}),
    (41, "ai_help", {}),
    (73, "check_meaning", {}),
    # with --cases data/eval/boundary_cases.jsonl
    (101, "cant_do", {"then": "ask if they are ready to begin."}),
    (106, "upset", {}),
    (109, "answer_question", {"facts": "You are the AI interviewer for this role and have no personal name.", "then": "ask if they are ready to begin."}),
    (110, "answer_question", {"facts": "You are the AI interviewer for this role and have no personal name.", "then": "ask if they are ready to begin."}),
]


# ---- decide engine: the model picks the action; gold actions come from the hand labels

ON_QUESTION = ["MOVE_ON", "FOLLOW_UP", "ASK_MORE", "WAIT_FOR_MORE", "ASK_SKIP", "REMIND_INTERVIEW", "REPEAT", "EXPLAIN",
               "CONFIRM_QUESTION", "RESUME_PREV",
               "TERM", "CHECK_MEANING", "CONFIRM_WORD", "ANSWER_THEM", "GIVE_TIME", "NOT_HEARD", "AUDIO", "AI_HELP",
               "STEER_BACK", "CONFIRM_END", "UPSET"]
BEFORE_Q1 = ["START", "NOT_READY", "ANSWER_THEM", "GIVE_TIME", "CHECK_MEANING", "NOT_HEARD", "AUDIO", "CONFIRM_END", "UPSET"]
END_CHECK = ["END", "CONTINUE", "CONFIRM_END", "ANSWER_THEM", "UPSET"]


def gold_actions(label: str, ctx: TurnContext, bot: str) -> set[str]:
    """What a good interviewer may do for a hand label, in this situation."""
    if ctx.bot_asked_end:
        return {"yes": {"END"}, "stop": {"END", "CONFIRM_END"}, "upset": {"UPSET"},
                "question": {"ANSWER_THEM"}}.get(label, {"CONTINUE"})
    if ctx.before_q1:
        return {"ready": {"START"}, "yes": {"START"}, "no": {"NOT_READY"}, "decline": {"NOT_READY"},
                "stop": {"CONFIRM_END"}, "upset": {"UPSET"}, "continuing": {"GIVE_TIME"}, "audio": {"AUDIO"},
                "unclear": {"NOT_HEARD", "CHECK_MEANING"}}.get(label, {"ANSWER_THEM"})
    moment = ctx.bot_offered_moment
    return {
        "answer": {"MOVE_ON", "FOLLOW_UP", "ASK_MORE", "WAIT_FOR_MORE"},
        "continuing": {"GIVE_TIME", "WAIT_FOR_MORE"},
        "ready": {"GIVE_TIME", "WAIT_FOR_MORE", "ANSWER_THEM"},
        "yes": {"GIVE_TIME", "WAIT_FOR_MORE"} if moment else {"GIVE_TIME", "WAIT_FOR_MORE", "ANSWER_THEM"},
        "no": {"ANSWER_THEM", "WAIT_FOR_MORE", "GIVE_TIME", "REMIND_INTERVIEW"},
        "repeat": {"REPEAT"},
        "rephrase": {"EXPLAIN"},
        "term_meaning": {"TERM"},
        "check_meaning": {"CONFIRM_QUESTION", "EXPLAIN", "CHECK_MEANING", "ANSWER_THEM"},
        "question": {"ANSWER_THEM"},
        "already_answered": {"MOVE_ON", "ANSWER_THEM", "ASK_SKIP", "REMIND_INTERVIEW"},
        "decline": {"ASK_SKIP"}, "dont_know": {"ASK_SKIP"}, "skip": {"ASK_SKIP"},
        "stop": {"CONFIRM_END"}, "go_back": {"GO_BACK", "ANSWER_THEM"},
        "correction": {"RESUME_PREV", "ANSWER_THEM", "WAIT_FOR_MORE", "CHECK_MEANING"},
        "ai_help": {"AI_HELP"}, "cant_do": {"ANSWER_THEM"}, "other": {"ANSWER_THEM"}, "upset": {"UPSET"},
        "audio": {"AUDIO"}, "off_topic": {"STEER_BACK"}, "unclear": {"NOT_HEARD", "CHECK_MEANING", "CHECK_REPLY"},
    }.get(label, {"ANSWER_THEM"})


def eval_decide(base, model, cases, plan, opts):
    from interview_bot.controller.decide import decide_messages, decide_schema

    p(f"\n=== {model} (decide engine)")
    chat(base, model, [{"role": "user", "content": "Hi"}], dict(opts, num_predict=1))
    ok, times, wrong, held, meanings, label_ok = 0, [], [], [], {}, 0
    n_q = len(plan.questions)
    for n, c in enumerate(cases, 1):
        conv, qtext = conversation(c, plan)
        ctx = context(c)
        bot = (c.get("bot") or "").lower()
        final = "anything you'd like to add" in bot
        if final:
            allowed = ["FINISH", "LISTEN_MORE", "ANSWER_THEM", "UPSET"]
        elif ctx.bot_asked_end:
            allowed = END_CHECK
        elif ctx.before_q1:
            allowed = BEFORE_Q1
        else:
            allowed = ON_QUESTION + (["GO_BACK"] if c["qid"] > 1 else [])
            if "move to the next question" in bot or "skip" in bot:
                allowed = allowed + ["SKIP", "CHECK_REPLY"]
            first_words = c["qid"] > 1 and not c.get("earlier") and not c.get("bot")
            if first_words:  # live: offered on their first words on a new question
                allowed = allowed + ["LATE_ANSWER"]
        if ctx.before_q1:
            state = "No question has been asked yet: you greeted them and asked if they are ready."
        else:
            state = (f'Current question {c["qid"]} of {n_q}: "{qtext}" They have {"" if ctx.answered else "not yet "}'
                     "given an answer to this question. Follow-ups left: 2. Repeats used: 0 of 3. "
                     "Clarifications left in the interview: 3. Refusals on this question: 0.")
            if not ctx.before_q1 and c["qid"] > 1 and not c.get("earlier") and not c.get("bot"):
                began = c.get("prev_began", "")
                state += (f' Previous question {c["qid"] - 1} (asked before this one): '
                          f'"{plan.questions[c["qid"] - 2].q}"' + (f' Their answer to it began: "{began}"' if began else ""))
        state += " About 15 of 20 minutes are left."
        if ctx.bot_asked_end:
            state += " You just asked if they want to end the interview."
        try:
            raw, secs, _ = chat(base, model, decide_messages(SYS, conv, state, allowed),
                                dict(opts, num_predict=90), fmt=decide_schema(allowed))
            data = json.loads(raw)
            action = data.get("action", "?")
            meanings[c["id"]] = f'{data.get("meaning", "")} [{data.get("label")}]'
            label_ok += data.get("label") in c["gold"]
            if action == "WAIT_FOR_MORE":
                held.append(c)
        except Exception as e:  # noqa: BLE001
            action, secs = f"error:{type(e).__name__}", 0.0
        times.append(secs)
        gold = {"LATE_ANSWER"} if c.get("late") else set()
        for g in ([] if c.get("late") else c["gold"]):
            gold |= ({"FINISH"} if g in ("no", "other") else {"LISTEN_MORE", "FINISH"} if g in ("yes", "answer")
                     else {"ANSWER_THEM"}) if final else gold_actions(g, ctx, bot)
        if action in gold:
            ok += 1
        else:
            wrong.append((c, action, gold))
        print(f"\r  {n}/{len(cases)} cases, {ok} right, last call {secs:.1f}s   ", end="", flush=True)
    print()
    acc = 100 * ok / len(cases)
    med = statistics.median(times)
    p(f"RIGHT ACTION {ok}/{len(cases)} = {acc:.0f}% | label {label_ok}/{len(cases)} = {100 * label_ok / len(cases):.0f}% "
      f"| per call median {med:.2f}s ({split()})")
    p(f"WAIT FOR MORE (bot stays silent 3 s): {len(held)}/{len(cases)}")
    for c, action, gold in wrong:
        p(f"  x #{c['id']} Q{c['qid']} {c['text'][:70]!r} -> {action} (ok: {'/'.join(sorted(gold))})"
          f"  | they mean: {meanings.get(c['id'], '')!r}")
    try:
        post(base, "/api/chat", {"model": model, "messages": [], "keep_alive": 0, "stream": False})
    except Exception:  # noqa: BLE001
        pass
    return (model, acc, med)


def main():
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["qwen3.5:4b", "qwen3.5:2b", "gemma4:e2b", "gemma4:e4b", "phi4-mini"])
    ap.add_argument("--gpu", action="store_true", help="allow the GPU (default: CPU only, like deployment)")
    ap.add_argument("--skip-talk", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N cases (a quick CPU speed check)")
    ap.add_argument("--cases", default=str(PROJECT_ROOT / "data" / "eval" / "understand_cases.jsonl"))
    ap.add_argument("--engine", default="two_step", choices=["two_step", "decide"],
                    help="two_step: the model labels, code decides | decide: the model picks the action (rules in its prompt)")
    ap.add_argument("--label-format", default="json", choices=["json", "compact"],
                    help="json: grammar-constrained JSON (~29 tokens) | compact: one short line (~8 tokens)")
    ap.add_argument("--plan", default=None,
                    help="the candidate folder or plan.json the cases come from (their question numbers refer to it); "
                         "default: Sai's plan for understand_cases, Rithika's for heldout_rithika, else INTERVIEW_PLAN_PATH")
    a = ap.parse_args()

    # The cases' question numbers refer to the interview they came from. Seen: with .env
    # pointing at another candidate, the 94 cases were run against the wrong questions and JD.
    name = Path(a.cases).stem
    default = {"understand_cases": "data/candidates/sai_shashank", "boundary_cases": "data/candidates/sai_shashank",
               "late_answer_cases": "data/candidates/sai_shashank",
               "heldout_rithika": "data/candidates/rithika"}.get(name)
    plan_path = Path(a.plan or default or s.INTERVIEW_PLAN_PATH)
    if not plan_path.is_absolute():
        plan_path = PROJECT_ROOT / plan_path
    if plan_path.is_dir():
        plan_path = plan_path / "plan.json"
    plan = QuestionPlan.load(plan_path)
    global SYS
    from interview_bot.controller.pipeline import jd_text_for
    SYS = system_prompt(plan.role, plan.first_name, len(plan.questions), round(s.INTERVIEW_TIME_LIMIT_SECS / 60),
                        jd_text_for(plan_path), label_format=a.label_format)
    if a.engine == "decide":
        from interview_bot.controller import decide as dmod
        SYS = dmod.system_prompt(plan.role, plan.first_name, len(plan.questions),
                                 round(s.INTERVIEW_TIME_LIMIT_SECS / 60), jd_text_for(plan_path))
    cases = [json.loads(x) for x in Path(a.cases).read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.limit:
        cases = cases[:: max(1, len(cases) // a.limit)][: a.limit]  # a spread of cases, not just the first ones
    cases.sort(key=lambda c: c["qid"])  # same question in a row: the system prompt stays cached
    # "localhost" costs ~2 s per new connection on Windows (IPv6 is tried first and
    # refused slowly); every urllib call is a new connection.
    base = s.OLLAMA_BASE_URL.rstrip("/").replace("://localhost", "://127.0.0.1")
    with urllib.request.urlopen(f"{base}/api/tags", timeout=10) as r:
        have = {m["name"] for m in json.load(r).get("models", [])}
    opts = {"temperature": 0.0, "num_ctx": 4096, "num_predict": 40}
    if not a.gpu:
        opts["num_gpu"] = 0
    p(f"Model eval {time.strftime('%Y-%m-%d %H:%M')} | {'GPU allowed' if a.gpu else 'CPU only'} | {len(cases)} cases "
      f"from {Path(a.cases).name} | plan {plan_path.parent.name} ({plan.candidate}) | label format {a.label_format}")
    lfmt = UNDERSTAND_SCHEMA if a.label_format == "json" else None
    opts_label = {"num_predict": 40 if a.label_format == "json" else 12}
    summary = []
    by_case = {c["id"]: c for c in cases}
    for model in a.models:
        if model not in have and f"{model}:latest" not in have:
            p(f"\n--- {model}: not installed, skipped (ollama pull {model})")
            continue
        if a.engine == "decide":
            summary.append(eval_decide(base, model, cases, plan, opts))
            continue
        think_ok = "thinking" in capabilities(base, model)
        p(f"\n=== {model}")
        t0 = time.monotonic()
        chat(base, model, [{"role": "user", "content": "Hi"}], dict(opts, num_predict=1), think_ok=think_ok)
        p(f"load: {time.monotonic() - t0:.1f}s")
        # Does Ollama reuse the prompt it just processed? (Hybrid / sliding-window models may not.)
        c0 = cases[0]
        conv0, q0 = conversation(c0, plan)
        m0 = label_messages(SYS, conv0, q0)
        lines = []
        for name, msgs, fmt in [
            ("first", m0, lfmt),
            ("exact same again", m0, lfmt),
            ("same start, new last message", m0[:-1] + [{"role": "user", "content": "Can you repeat the question?"}], lfmt),
            ("same, no output format", m0[:-1] + [{"role": "user", "content": "Can you repeat the question please?"}], None),
        ]:
            _, secs, _ = chat(base, model, msgs, dict(opts, **opts_label), fmt=fmt)
            lines.append(f"  {name:30s} {secs:5.2f}s = {split()}")
        p("timing test (Ollama's own numbers):\n" + "\n".join(lines))
        ok, moves_ok, times, wrong, labels, newtok, held, bad_form = 0, 0, [], [], Counter(), [], [], []
        for n, c in enumerate(cases, 1):
            conv, qtext = conversation(c, plan)
            try:
                raw, secs, ntok = chat(base, model, label_messages(SYS, conv, qtext), dict(opts, **opts_label),
                                       fmt=lfmt, think_ok=think_ok)
                newtok.append((LAST.get("prompt_eval_duration", 0) + LAST.get("eval_duration", 0)) / 1e9)
                parsed = parse_label(raw, a.label_format)
                if parsed is None:
                    bad_form.append(raw)
                    label = "?"  # the live bot would ask again with the JSON grammar
                else:
                    label = parsed[0]
                    if parsed[4] is False:
                        held.append(c)
            except Exception as e:  # noqa: BLE001
                label, secs = f"error:{type(e).__name__}", 0.0
            times.append(secs)
            labels[label] += 1
            ctx = context(c)
            gold_moves = {decide_move(g, ctx) for g in c["gold"]}
            move_ok = decide_move(label, ctx) in gold_moves
            moves_ok += move_ok
            if label in c["gold"]:
                ok += 1
            else:
                wrong.append((c, label, move_ok))
            print(f"\r  {n}/{len(cases)} cases, {ok} right, last call {secs:.1f}s   ", end="", flush=True)
        print()
        acc = 100 * ok / len(cases)
        med, p90 = statistics.median(times), sorted(times)[int(0.9 * len(times))]
        tok = statistics.median(newtok) if newtok else 0
        p(f"UNDERSTAND accuracy {ok}/{len(cases)} = {acc:.0f}% | per call median {med:.2f}s, p90 {p90:.2f}s "
          f"(model's own time: median {tok:.2f}s)")
        p(f"RIGHT MOVE (what the bot actually does after code applies the label): {moves_ok}/{len(cases)} = "
          f"{100 * moves_ok / len(cases):.0f}%")
        # "finished": false makes the bot wait 3 s more in silence. Most of these cases are
        # finished turns, so a high number here means extra delay in real interviews.
        if bad_form:
            p(f"NOT IN THE ASKED FORM: {len(bad_form)} (live: asked again as JSON) e.g. {bad_form[0][:60]!r}")
        p(f"WAIT FOR MORE (model says they stopped mid-thought): {len(held)}/{len(cases)} = "
          f"{100 * len(held) / len(cases):.0f}% of turns would wait 3 s more")
        for c in held[:8]:
            p(f"  ~ #{c['id']} {c['text'][-60:]!r}")
        for c, label, move_ok in wrong:
            tag = "harmless, same move" if move_ok else f"WRONG MOVE: {decide_move(label, context(c))}"
            p(f"  x #{c['id']} Q{c['qid']} {c['text'][:70]!r} -> {label} (expected {'/'.join(c['gold'])}) [{tag}]")
        talk_times = []
        if not a.skip_talk:
            p("TALK samples:")
            for cid, move, extra in TALK_SAMPLES:
                c = by_case.get(cid)
                if not c:
                    continue
                conv, qtext = conversation(c, plan)
                try:
                    reply, secs, _ = chat(base, model, say_messages(SYS, conv, move, question=qtext or "", **extra),
                                          dict(opts, num_predict=60, temperature=0.4), think_ok=think_ok)
                except Exception as e:  # noqa: BLE001
                    reply, secs = f"(error {type(e).__name__})", 0.0
                talk_times.append(secs)
                p(f"  [{move}] candidate: {c['text'][:60]!r}\n      -> {reply.strip()!r} ({secs:.2f}s)")
        summary.append((model, acc, 100 * moves_ok / len(cases), med, statistics.median(talk_times) if talk_times else 0.0))
        try:
            post(base, "/api/chat", {"model": model, "messages": [], "keep_alive": 0, "stream": False})
        except Exception:  # noqa: BLE001
            pass

    p("\n=== summary (understand accuracy, seconds per call)")
    if a.engine == "decide":
        for model, acc, med in summary:
            p(f"  {model:14s} right action {acc:4.0f}% | {med:5.2f}s per call")
        summary = []
    for model, acc, macc, med, tmed in sorted(summary, key=lambda x: -x[2]):
        p(f"  {model:14s} label {acc:4.0f}% | right move {macc:4.0f}% | {med:5.2f}s per call | talk {tmed:5.2f}s")
    path = PROJECT_ROOT / "logs" / f"eval_models_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    path.write_text("\n".join(OUT), encoding="utf-8")
    p(f"\nSaved {path.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
