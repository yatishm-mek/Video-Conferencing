"""Two-step turns: (1) UNDERSTAND what the candidate meant (one label, constrained
JSON), (2) code picks the move from the label + interview state (repeat and
clarification limits, stop ladder), (3) TALK: the model writes one short line
for that move. Both calls see the conversation as real chat turns.

Small models label much more reliably than they follow a long rulebook, and the
talk call only has to do one thing, so its prompt stays short."""

from __future__ import annotations

from dataclasses import dataclass, field

LABELS: dict[str, str] = {
    "ready": "before the first question: agrees to start ('yes', 'okay', 'let's go', 'I am ready')",
    "answer": "says something about the current question: a full or partial answer, even a short one like 'I have no work experience'",
    "continuing": "asks for time or is in the middle of a thought ('let me think', 'one second')",
    "yes": "agrees with or accepts the interviewer's last question, in any wording, including a question that means yes, or extra words ('yeah', 'okay, yes')",
    "no": "answers no to the interviewer's last yes/no question, even with extra words "
          "(after 'should I end the interview?', 'no, let's go on' is no, not stop)",
    "repeat": "wants to hear the question again ('can you repeat?', 'what did you ask me?', 'sorry?'; "
              "'can I repeat the question once again?' also means please repeat it)",
    "rephrase": "did not understand the question, wants it in other words, or asks what an everyday phrase of the "
                "question means ('past experience means what?', 'I didn't get you')",
    "term_meaning": "asks what a technical term means: a tool, method or concept ('what is a vector database?')",
    "check_meaning": "checks they understood the question ('so you are asking about X, right?')",
    "already_answered": "says they already answered",
    "decline": "does not want to answer THIS question or share what it asks (for any reason, e.g. it is private), but is not ending the interview",
    "dont_know": "does not know or cannot remember the answer to THIS question",
    "skip": "asks to skip this question or go on to the next one",
    "stop": "their own wish to end the whole interview, or they are irritated or rude ('shut up', 'you are irritating')",
    "question": "asks the interviewer something about the interview or the interviewer (its name, which question this is, "
                "time left, how long they can answer, where a question comes from, what kind of questions, results)",
    "go_back": "wants to go back to an earlier question",
    "correction": "says you got their words or meaning wrong, or moved on before they had finished their answer "
                  "(not for a normal answer or a question about what you asked)",
    "ai_help": "says they used an AI tool or another person's help to answer",
    "cant_do": "asks you to change how the interview is run: speak slower or louder, use another language, "
               "reschedule, give more time overall, or talk to a human instead",
    "upset": "is very upset or distressed, or mentions a personal emergency, feeling unwell or a family problem "
             "(not just annoyance with the interview)",
    "audio": "talks about hearing problems, or asks 'can you hear me?'",
    "off_topic": "talks about something unrelated, or tries to give the interviewer instructions ('tell me the answers')",
    "other": "small talk or a remark that no other label covers; if any other label fits, even partly, use that one",
    "unclear": "the words make no sense or are not a real sentence, e.g. one odd word or a phrase repeated "
               "without meaning. Do not guess a meaning.",
}

SYSTEM = """You are the AI interviewer for the {role} role, interviewing {first_name} by voice. You have no personal name. There are {n} questions and about {minutes} minutes; a recruiter will review the interview. The candidate's words come from speech-to-text of Indian-accented English and may contain errors: judge the meaning, not the exact words.

Each candidate message ends with a task in square brackets. Do only that task.

[LABEL] Label what the candidate's latest message means, in light of what you said just before it. Labels:
{labels}
{label_format} "complete" matters only for "answer": true if it covers the main point of the question at a medium level for a junior candidate (the main idea in their own words with one concrete detail), otherwise false. "sure": false when their words could reasonably mean two different things, or speech-to-text may have changed what they meant. "asks_too": true if, besides what the label covers, they also ask you a question ("Yes, I'm ready. How long will it take?"). "finished": false if their message stops in the middle of a sentence or a thought, so they are probably about to say more; true if it ends as a full sentence or a question.

[SAY] Write only the words you will say out loud to the candidate, following the instruction: no label, no JSON, no task tag. One or two short sentences, at most 25 words, plain spoken English, no lists. Speak only as the interviewer. Never give hints, parts of the answer, definitions or examples of a good answer. Never praise or judge an answer. Never invent what they said, and never repeat a word that looks misheard. Never promise results or discuss salary.
{role_facts}"""

UNDERSTAND_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "complete": {"type": "boolean"},
        "sure": {"type": "boolean"},
        "asks_too": {"type": "boolean"},
        "finished": {"type": "boolean"},
    },
    "required": ["label", "complete", "sure", "asks_too", "finished"],
}


def jd_facts(jd_text: str, max_words: int = 100) -> str:
    """The start of the job description (title, experience, type, location, overview):
    the only role facts the bot may share. Kept short: it sits in the system prompt,
    which Ollama reads once per interview."""
    out: list[str] = []
    n = 0
    for line in jd_text.splitlines():
        words = line.split()
        if not words:
            continue
        if n + len(words) > max_words:
            words = words[: max(0, max_words - n)]
            if len(words) >= 5:
                out.append(" ".join(words) + " ...")
            break
        out.append(" ".join(words))
        n += len(words)
    while out and len(out[-1].split()) < 4 and ":" not in out[-1]:
        out.pop()  # a heading with nothing under it ("Key Responsibilities")
    return "\n".join(out)


LABEL_FORMATS = {
    # JSON, constrained by Ollama's grammar: never malformed, but ~29 tokens to write
    # (2 s on a laptop CPU at ~14 tokens/s).
    "json": 'Return JSON: {{"label": "<one label>", "complete": true or false, "sure": true or false, '
            '"asks_too": true or false, "finished": true or false}}.',
    # One short line: ~8 tokens. Same meaning, a quarter of the writing time on CPU.
    "compact": "Reply with one line only, five words separated by spaces: the label, then y or n for each of "
               "complete, sure, asks_too and finished, in that order (the form: <label> <y/n> <y/n> <y/n> <y/n>).",
}


def parse_label(raw: str, fmt: str = "json") -> tuple[str, bool, bool, bool, bool] | None:
    """The model's understanding -> (label, complete, sure, asks_too, finished), or None."""
    raw = (raw or "").strip()
    if fmt == "compact" or not raw.startswith("{"):
        words = re.findall(r"[a-z_]+", raw.lower().split("\n")[0])
        if not words or words[0] not in LABELS:
            return None
        flags = [w in ("y", "yes", "true") for w in words[1:5]] + [None] * 4
        complete, sure, asks_too, finished = flags[:4]
        return (words[0], bool(complete), True if sure is None else sure, bool(asks_too),
                True if finished is None else finished)
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    label = str(data.get("label", ""))
    if label not in LABELS:
        return None
    return (label, bool(data.get("complete", False)), bool(data.get("sure", True)),
            bool(data.get("asks_too", False)), bool(data.get("finished", True)))


def system_prompt(role: str, first_name: str, n_questions: int, minutes: int, jd_text: str = "",
                  label_format: str = "json") -> str:
    """One system prompt for the whole interview, shared by both calls: Ollama
    reuses it (and the conversation) instead of reading it again each call."""
    facts = jd_facts(jd_text) if jd_text else ""
    role_facts = (
        "\nRole facts from the job description. These are the only facts about the role or company you may share; "
        "for HR matters not written here (salary, leave, benefits, the offer, joining date, results, next steps, the "
        f"team) say the recruiter can help with that:\n<job_description>\n{facts}\n</job_description>" if facts else
        "\nYou have no details about the role or company; for those and for HR matters (salary, leave, benefits, the "
        "offer, results, next steps) say the recruiter can help with that."
    )
    return SYSTEM.format(
        role=role or "this", first_name=first_name or "the candidate", n=n_questions, minutes=minutes,
        labels="\n".join(f"- {k}: {v}" for k, v in LABELS.items()), role_facts=role_facts,
        label_format=LABEL_FORMATS[label_format].replace("{{", "{").replace("}}", "}"),
    )


MOVES: dict[str, str] = {
    "bridge": "Say one short neutral sentence (at most 12 words) that reflects the main thing they just said, in the form \"Okay, so you <one main thing, in their words>.\" Don't list everything. Use only things they said. No question. The next question follows right after you.",
    "follow_up": "The question was: \"{question}\". If they talked about a different project or topic than this question asks about, say politely that you'd like to hear about what the question asks, and ask it again in a few words. Otherwise ask one short follow-up question about what they said, on the part of the question they have not covered yet. End with a question mark. Do not hint at the answer.",
    "final_reply": "You asked if they want to add anything before you finish, and they replied. In one short sentence thank them; if they added something, acknowledge it briefly without judging it. Do not ask anything.",
    "answer_question": "Reply naturally and kindly to what they just said, in one sentence: answer it if it is a question, acknowledge it if it is a remark or a feeling. Use only these facts and the role facts you were given: {facts} Questions about this interview itself (how long, how much time per answer, who is taking it, where the questions come from) are yours: say this is an AI interview and answer from these facts, never send them to the recruiter. Only HR matters that are not in the facts (salary, leave, benefits, the offer, results, next steps) go to the recruiter: say the recruiter can help with that. Never guess. Never promise anything you cannot do (you cannot change your speed, language or the schedule). Then {then}",
    "cant_do": "They asked for something you cannot change in this interview. In one honest, kind sentence say you can't change that here, and if it is about the schedule or anything outside the interview, that the recruiter can help with that. Promise nothing. Then {then}",
    "upset": "They seem upset or have a personal problem. Say kindly that you're sorry to hear that, and ask if they would like to stop the interview now. They can answer yes or no. Nothing else.",
    "rephrase": "Say the current question again in simpler words, with the same meaning and the same technical terms, and nothing else. The current question is: \"{question}\"",
    "clarify": "If they asked what an everyday phrase of the question means, say in one sentence what kind of answer you want, without hints. If they asked what a technical term means, say kindly that you can't explain terms during the interview and they can answer as they understand it. The current question is: \"{question}\"",
    "check_meaning": "They checked whether they understood the question. The question is: \"{question}\". If they understood it, say yes briefly and invite them to answer. If not, say what it asks in other words, without hints.",
    "not_yet": "Say kindly that you haven't heard their answer to this question yet, and ask them to answer it briefly.",
    "decline": "Say kindly that this is an AI interview and you'd like to hear their answer, even briefly or in general terms if something is private; if they still prefer not to, they can tell you and you'll move on.",
    "repeat_limit": "Say kindly that you have already repeated this question, ask them to answer it as they understand it, and mention that about {left} minutes are left.",
    "limit_time": "Say kindly that you need to keep the interview moving: about {left} of the {total} minutes are left. Ask for their best answer now, or they can say so and you'll move on.",
    "confirm_end": "Say that's fine, and ask if they would like to end the interview now. They should answer yes or no.",
    "confirm_end_last": "Ask once more if they want to end the interview now, and say this is the last time you will ask.",
    "ai_help": "Thank them for being honest, and ask them to answer in their own words, from what they understand.",
    "off_topic": "Steer them back to the question kindly, mention that about {left} of the {total} minutes are left, and ask them to answer it.",
    "audio": "Say you can hear them, and ask them to go on.",
    "go_on": "Acknowledge them in a few words and ask them to go on with their answer.",
    "correction": "Apologise briefly for the mix-up and ask them to go on.",
    "go_back_no": "Say kindly that you can't go back to earlier questions, and ask them to answer this one.",
    "continuing": "Say a few words so they know they can take their time.",
    "moment": "They have been quiet since your question. Ask kindly if they would like a moment to think.",
    "long_silence": "They have been quiet for a long time. Offer to repeat the question or move on.",
    "ready_again": "They have been quiet. Ask kindly if they are ready to begin.",
    "confirm_meaning": "You are not sure what they meant. Ask briefly to check, in your own words: do they mean {meaning}? They can answer yes or no.",
    "confirm_skip": "They asked to skip. Ask them to confirm, naming the question by its number ({n}) and its topic in a few words, e.g. \"Okay, you'd like to skip question {n}, about ..., right?\" The question is: \"{question}\"",
    "gentle_reminder": "They asked for a moment and have been quiet for a while. In a few words, gently tell them to take their time and {then}.",
    "not_ready": "They said they are not ready. In two short sentences: say that's okay and that if they are feeling nervous you can give them a moment, and that they can say 'ready' when they want to start, or tell you if they'd like to end the interview. Ask no other question.",
    "end_silence": "They have not answered whether to end the interview. Ask again briefly: should you end it now, yes or no?",
    "no_more_help": "Say kindly that this is an interview, so you can't explain it further, that they can answer as they understand it, and that about {left} minutes are left.",
    # decide engine (decide.py)
    "remind_interview": "Say kindly that this is an interview with a time limit: about {left} of the {total} minutes are left. Ask them to tell you what they have understood of the question and answer with that, even briefly.",
    "ask_skip": "Ask them to confirm if they want to move to the next question, naming the current one by its number ({n}) and topic in a few words. The question is: \"{question}\"",
    "refuse_end": "Say kindly that it seems they don't want to continue this interview, and ask if you should end it now. They can answer yes or no.",
    "explain": "They didn't understand the question. Say it again in other, simpler words with the same meaning and no hints. The current question is: \"{question}\"",
    "term": "Say kindly that you can't explain technical terms during the interview, and that they can answer as they understand it.",
    "check_meaning_free": "You are not sure what they meant. Ask briefly, using their own words, whether they meant what you think; they can answer yes or no. Do not guess anything new.",
    "confirm_word": "One important technical word they said may have been misheard. Ask only \"Did you say <the word you think they said>?\", where the word sounds like what was transcribed.",
    "not_heard": "Say you didn't catch that and ask them to say it again.",
}


@dataclass
class Conversation:
    """The interview so far as chat turns: ("bot", text) / ("candidate", text)."""
    turns: list[tuple[str, str]] = field(default_factory=list)

    def chat(self, last_n: int = 10) -> list[dict]:
        msgs: list[dict] = []
        for who, text in self.turns[-last_n:]:
            role = "assistant" if who == "bot" else "user"
            if msgs and msgs[-1]["role"] == role:  # merge consecutive lines from one side
                msgs[-1]["content"] += " " + text
            else:
                msgs.append({"role": role, "content": text})
        if msgs and msgs[0]["role"] == "user":  # chat templates expect the bot's line first
            msgs.insert(0, {"role": "assistant", "content": "(interview in progress)"})
        if not msgs or msgs[-1]["role"] == "assistant":
            msgs.append({"role": "user", "content": "(the candidate is silent)"})
        return msgs


def _with_task(system: str, conv: Conversation, task: str) -> list[dict]:
    msgs = conv.chat()
    msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + "\n\n" + task}
    return [{"role": "system", "content": system}, *msgs]


def label_messages(system: str, conv: Conversation, current_question: str | None) -> list[dict]:
    where = (f'The current question is: "{current_question}"' if current_question else
             "No question has been asked yet: you greeted them and asked if they are ready.")
    return _with_task(system, conv, f"[LABEL] {where}")


def say_messages(system: str, conv: Conversation, move: str, **fmt) -> list[dict]:
    instruction = MOVES[move].format(**{"facts": "", "then": "", "question": "", "left": "", "total": "", "n": "",
                                        "meaning": "", **fmt})
    return _with_task(system, conv, f"[SAY] {instruction} (Plain spoken words only.)")


# What a label means, said back to the candidate when the model is not sure ("do you mean ...?").
MEANINGS: dict[str, str] = {
    "ready": "that they are ready to start",
    "decline": "that they would rather not share those details",
    "dont_know": "that they don't know the answer to this one",
    "already_answered": "that they have already answered this question",
    "go_back": "that they want to go back to an earlier question",
    "correction": "that you misunderstood them",
    "skip": "that they want to skip this question",
    "stop": "that they want to end the interview",
}


@dataclass
class TurnContext:
    """What the code knows when it turns a label into a move."""
    before_q1: bool = False
    bot_asked_end: bool = False      # the bot's last line asked "should I end the interview?"
    bot_offered_moment: bool = False  # the bot's last line asked "would you like a moment?"
    answered: bool = False            # this question already got an answer
    words: int = 0                    # words in the candidate's turn


def decide_move(label: str, ctx: TurnContext) -> str:
    """Label + situation -> what the bot does. Limits (repeats, clarifications, the
    stop ladder) are applied on top of this by the controller."""
    if label in ("unclear", "correction") and ctx.words >= 15 and not ctx.before_q1 and not ctx.bot_asked_end:
        # A long turn is content: never noise (live: an answer with "chinking" in it), and never
        # just "you misunderstood me" (live: a full answer got "I apologize for the confusion").
        # If it does correct the bot, the reply to the answer reflects what they now said.
        label = "answer"
    if label == "upset":
        return "upset"  # a personal problem comes first, whatever was asked
    if ctx.bot_asked_end:
        # Only a clear yes ends the interview; anything else carries on.
        return {"yes": "end", "stop": "confirm_end_again"}.get(label, "continue")
    if label == "cant_do":
        return "cant_do"  # same reply before and during the questions
    if ctx.before_q1:
        if label in ("ready", "yes"):
            return "start"
        if label == "stop":
            return "confirm_end"
        return "reply_then_ready"  # answer what they said, then ask if they are ready
    if ctx.bot_offered_moment and label in ("yes", "continuing", "ready"):
        return "listen"
    return {
        "answer": "answer",            # complete -> bridge + next question, else a follow-up
        "continuing": "listen",
        "ready": "go_on",
        "yes": "go_on",
        "no": "go_on",
        "audio": "go_on",
        "repeat": "repeat",
        "rephrase": "clarify",         # one move: explains an everyday phrase, declines a technical term
        "term_meaning": "clarify",
        "check_meaning": "check_meaning",
        "already_answered": "next" if ctx.answered else "not_yet",
        "decline": "decline",
        "dont_know": "move_on",
        "skip": "move_on",
        "stop": "confirm_end",
        "question": "answer_question",
        "other": "answer_question",  # beyond our list: the model still replies, then steers back
        "go_back": "go_back",
        "correction": "correction",
        "ai_help": "ai_help",
        "off_topic": "off_topic",
        "unclear": "unclear",
    }.get(label, "unclear")


@dataclass
class Decision:
    """What the bot does this turn: the action for the controller, and either a
    fixed short line or a [SAY] move for the model to word."""
    action: str
    move: str = ""
    text: str = ""
    fmt: dict = field(default_factory=dict)
    first_only: bool = False  # a bridge: one sentence, never a question
    label: str = ""
    fallback: str = ""  # said if the model's line is dropped


# ------------------------------------------------------------------ the turn runner

import asyncio  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

from loguru import logger  # noqa: E402
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame  # noqa: E402

from interview_bot.controller.natural import (  # noqa: E402
    NO_MORE_HELP,
    _said_before,
    StaleTurn,
    TurnOutcome,
    clean_for_speech,
    guard_sentence,
)

_SENT = re.compile(r"(?<=[.!?])\s+")
_NUM = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
_QNUM = re.compile(rf"\bquestion {_NUM} of {_NUM}\b", re.I)
_MOVE_TALK = re.compile(r"\b(move on|moving on|next question)\b", re.I)
_NO_MOVE_TALK = {"check_meaning", "check_meaning_free", "answer_question", "explain", "continuing", "go_on", "term",
                 "not_heard", "audio", "confirm_word", "clarify", "rephrase", "correction"}
BRIDGE_MAX_WORDS = 16  # a bridge is said before the next question: long ones are slow to voice
_TAG = re.compile(r"^\s*\[(SAY|LABEL)\]\s*", re.I)
_FALLBACK = {
    "FOLLOW_UP": "Could you tell me a bit more about that?",
    "REPHRASE": "Let me put it simply: please answer it as you understand it.",
    "CLARIFY": "Please answer it as you understand it.",
    "CONFIRM_END": "That's fine. Would you like me to end the interview now? Please say yes or no.",
}


class TwoStepRunner:
    """Turn = UNDERSTAND (label, JSON) -> controller decides -> SAY (streamed)."""

    def __init__(self, llm, cfg):
        self._llm, self._cfg = llm, cfg

    async def prewarm(self, messages: list[dict]):
        t0 = time.monotonic()
        try:
            await self._llm.chat(self._cfg.model, messages, think=False, max_tokens=1, temperature=0.0)
            logger.info(f"[llm] model ready for this interview ({time.monotonic() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[llm] prewarm failed ({e or type(e).__name__}); first reply may be slow")

    async def _decide(self, req, ctrl, conv, t0):
        """Step 1: understand the candidate -> the controller's Decision (None: the model failed)."""
        cfg = self._cfg
        label, complete, sure, asks_too, finished = "", False, True, False, True
        if not req.event:
            try:
                msgs = label_messages(ctrl.ts_system, conv, req.question_text or None)
                compact = getattr(cfg, "label_format", "json") == "compact"
                raw = await asyncio.wait_for(
                    self._llm.chat(cfg.model, msgs, think=False, temperature=0.0, max_tokens=12 if compact else 40,
                                   **({} if compact else {"format": UNDERSTAND_SCHEMA})),
                    timeout=cfg.turn_timeout_secs,
                )
                parsed = parse_label(raw, "compact" if compact else "json")
                if parsed is None and compact:
                    # Not in the expected form: ask again with the grammar, which can't fail.
                    logger.info(f"[turn] {req.turn_id} label line not understood ({raw!r}); asking again as JSON")
                    raw = await asyncio.wait_for(
                        self._llm.chat(cfg.model, msgs, think=False, temperature=0.0, max_tokens=40,
                                       format=UNDERSTAND_SCHEMA),
                        timeout=cfg.turn_timeout_secs,
                    )
                    parsed = parse_label(raw, "json")
                if parsed is None:
                    raise ValueError(f"no label in {raw!r}")
                label, complete, sure, asks_too, finished = parsed
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[turn] {req.turn_id} understand failed: {e or type(e).__name__}")
                return None
        return ctrl.decide_two_step(req.turn_id, label or "", complete, sure, asks_too=asks_too, finished=finished)

    async def run(self, req, push, ctrl, on_speech, voice=None) -> TurnOutcome | None:
        t0 = time.monotonic()
        conv = Conversation(list(req.conversation))
        d = await self._decide(req, ctrl, conv, t0)
        if d is False:
            return None  # the model failed: the caller falls back
        if d is None:
            if ctrl.pending_turn_id != req.turn_id:
                raise StaleTurn()
            return None
        t_label = time.monotonic() - t0
        label = d.label
        if d.action == "HOLD":
            # They stopped mid-thought: say nothing and wait for the rest.
            logger.info(f"[turn] {req.turn_id} {label} but unfinished -> wait silently for more "
                        f"(understand {time.monotonic() - t0:.2f}s)")
            return TurnOutcome(req.turn_id, "HOLD", (), "", answer=None)

        spoken: list[str] = []
        replaced: list[tuple[str, str]] = []
        started = False
        lat = {"understand": t_label}

        async def emit(text: str):
            nonlocal started
            text = clean_for_speech(text).strip()
            if not text:
                return
            if ctrl.pending_turn_id != req.turn_id:
                raise StaleTurn()
            if not started:
                on_speech()
                await push(LLMFullResponseStartFrame())
                started = True
                lat["first_speech"] = time.monotonic() - t0
            words = text.split()
            # (Not for a 1-2 word lead-in: "Okay," alone played, then a gap while the rest was
            # voiced - "it makes my heart drop, did the bot fail?")
            cut = next((i for i, w in enumerate(words[:-4]) if w.endswith(",") and i >= 2), None) if len(spoken) == 0 else None
            if cut is not None and len(words) > 8:
                # Close the first clause as its own piece so the voice starts on it now
                # (long first sentences made the first audio 1-3 s late).
                await push(LLMTextFrame(" ".join(words[: cut + 1]) + " "))
                await push(LLMFullResponseEndFrame())
                await push(LLMFullResponseStartFrame())
                await push(LLMTextFrame(" ".join(words[cut + 1:]) + " "))
            else:
                await push(LLMTextFrame(text + " "))
            spoken.append(text)

        if d.text:
            await emit(d.text)
        elif d.move:
            await self._talk(req, conv, ctrl, d, emit, spoken, replaced)
            if not spoken and replaced and replaced[-1][1] == "label instead of words":
                # Seen in eval: a [SAY] call answered with the [LABEL] JSON. Ask once more.
                await self._talk(req, conv, ctrl, d, emit, spoken, replaced, retry=True)
            if not spoken:
                # Live: a follow-up that hinted ("how did you handle missing values?") was replaced
                # with "This is an interview, so I can't explain it further" - they had asked nothing.
                hint = d.move in ("clarify", "rephrase", "check_meaning", "answer_question") and any(
                    w.startswith(("explains a term", "answer words")) for _, w in replaced)
                calm = d.move in ("moment", "continuing", "long_silence") and "Take your time."
                fb = NO_MORE_HELP if hint else calm or d.fallback or _FALLBACK.get(
                    d.action, "" if d.action in ("NEXT_QUESTION", "REPEAT_QUESTION", "GO_BACK") else "Okay.")
                if d.action == "REPHRASE" and not hint:
                    d.action, fb = "REPEAT_QUESTION", "Let me repeat it."
                await emit(fb)
        keep_open = started and d.action in ("NEXT_QUESTION", "REPEAT_QUESTION", "GO_BACK")
        if started and not keep_open:
            await push(LLMFullResponseEndFrame())
        lat["total"] = time.monotonic() - t0
        said = " ".join(spoken)
        logger.info(
            f"[turn] {req.turn_id} {d.label or 'event'} -> {d.move or 'fixed'} -> {d.action} | understand {t_label:.2f}s, "
            f"first speech {lat.get('first_speech', 0):.2f}s, total {lat['total']:.2f}s -> {said!r}"
        )
        return TurnOutcome(
            req.turn_id, d.action, (), said, answer=(d.label == "answer") if d.label else None,
            replaced=replaced, sentences=len(spoken), latency=lat, open_response=keep_open,
        )

    async def _talk(self, req, conv, ctrl, d: Decision, emit, spoken, replaced, retry: bool = False):
        """Stream the [SAY] reply sentence by sentence through the guards."""
        cfg = self._cfg
        msgs = say_messages(ctrl.ts_system, conv, d.move, **d.fmt)
        if retry:
            msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + " Do not return a label or JSON."}
        buf = ""
        stream = None
        try:
            stream = self._llm.chat_stream(cfg.model, msgs, think=False, temperature=cfg.temperature, max_tokens=70)
            async for piece in stream:
                buf += piece
                if buf.lstrip().startswith("{") or '"label"' in buf:
                    replaced.append((buf.strip(), "label instead of words"))
                    logger.info(f"[guard] the reply was a label, not words: {buf.strip()!r}")
                    return
                parts = _SENT.split(buf)
                for sentence in parts[:-1]:
                    if await self._say(sentence, d, req, emit, spoken, replaced):
                        break
                else:
                    buf = parts[-1]
                    continue
                buf = ""
                break  # the reply is complete (a bridge: one sentence)
            if buf.strip():
                await self._say(buf, d, req, emit, spoken, replaced)
        except StaleTurn:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[turn] {req.turn_id} say failed: {e or type(e).__name__}")
        finally:
            if stream is not None:
                try:
                    await stream.aclose()  # stop generating once the reply is out
                except Exception:  # noqa: BLE001
                    pass

    async def _say(self, sentence, d: Decision, req, emit, spoken, replaced) -> bool:
        """Guard and speak one sentence. Returns True when the reply is complete."""
        sentence = _TAG.sub("", sentence).strip()  # seen: "[SAY] Could you tell me ...?"
        if re.sub(r"[^a-z_ ]", "", sentence.lower()).strip() in LABELS:
            # Live: a clarify reply came out as just '"rephrase"'.
            replaced.append((sentence, "label instead of words"))
            logger.info(f"[guard] the reply was a label, not words: {sentence!r}")
            return False
        if d.action == "FOLLOW_UP" and sentence.rstrip().endswith("?") and _said_before(sentence, req.recent_bot):
            # Live: the same follow-up was asked twice in a row ("Could you tell me a bit
            # (more) about the kind of work you have been doing recently?").
            replaced.append((sentence, "same follow-up again"))
            logger.info(f"[guard] same follow-up again: {sentence!r}")
            await emit("Please go on with your answer.")
            return True
        qpos = d.fmt.get("_qpos") if d.fmt else None
        if qpos:
            # Live: on question 6 it said "This is question 2 of 6" twice and confused her.
            fixed = _QNUM.sub(lambda m: f"question {qpos[0]} of {qpos[1]}", sentence)
            if fixed != sentence:
                logger.info(f"[guard] question number fixed: {sentence!r} -> {fixed!r}")
                sentence = fixed
        text, why = guard_sentence(sentence, req)
        if why:
            replaced.append((sentence, why))
            logger.info(f"[guard] {why}: {sentence!r} -> {text!r}")
        if not text:
            return False
        if d.action != "NEXT_QUESTION" and d.move in _NO_MOVE_TALK and _MOVE_TALK.search(text):
            # Live: "Let's move on to the next question." in a reply that did not move on.
            replaced.append((text, "said it moves on but doesn't"))
            logger.info(f"[guard] says it moves on but the action doesn't: {text!r}")
            return False
        if d.first_only:
            if text.rstrip().endswith("?"):
                replaced.append((text, "question in the bridge"))
                return False
            if len(text.split()) > BRIDGE_MAX_WORDS:
                # Live: a 35-word bridge listing everything she said took Kokoro ~6 s; she said
                # "Yeah?" into the silence, the move was taken back, and it looped 4 times.
                replaced.append((text, "bridge too long"))
                logger.info(f"[guard] bridge too long ({len(text.split())} words): {text!r}")
                text = "Okay, thank you."
            await emit(text)
            return True
        await emit(text)
        return len(spoken) >= 2
