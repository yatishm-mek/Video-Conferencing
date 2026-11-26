"""Fast, rule-based reading of short candidate replies.

No LLM: these run on every turn in microseconds, on GPU or CPU alike, and
decide nothing by themselves. The controller uses them for the moments that
need an instant answer ("can you repeat?", "give me a minute", "are you a
bot?"). From Phase 4 the evaluator LLM also returns an intent; the controller
takes the rule result when it fires and the evaluator's otherwise.

Every rule only applies to short utterances, so a long answer that happens to
contain "repeat" or "not sure" is still treated as an answer.
"""

from __future__ import annotations

import re

from interview_bot.services.stt import normalize

SHORT = 14  # words; longer replies are answers, whatever they contain


def _words(text: str) -> list[str]:
    return normalize(text).replace("-", " ").split()


def _has(text: str, patterns: list[str]) -> bool:
    t = " " + " ".join(_words(text)) + " "
    return any(re.search(p, t) for p in patterns)


# Patterns run on " " + normalized words + " ", so " x " is a whole word and
# "^ x" means the reply starts with x.
_REPEAT = [
    r" repeat", r" say (that|it) again", r" come again", r" pardon", r" one more time",
    r" didn'?t (get|catch|hear|understand) ", r" did not (get|catch|hear|understand) ",
    r" (can|could) you rephrase", r" what (was|is) the question", r" sorry what ",
    r"^ (what|sorry|huh|hmm what) $", r" (question|that) (is|was) not clear", r" not clear to me",
    r" i missed (that|it|the question)",
]
# "The previous question" means the one before the current one. ("The last
# question" usually means the one just asked, so it stays a plain repeat.)
_PREVIOUS = [
    r" (previous|earlier|prior) question", r" question before", r" go back to (the )?(previous|earlier|last|that)",
    r" go back ", r" before question",
]
# Asking permission to answer: "I remembered, can I answer it now?"
_GO_ON = [
    r" can i (answer|try|say it|add|continue|go ahead)", r" (let me|may i) (answer|try|add|continue)",
    r" i (just )?remembered (the answer|it)", r" i got the answer", r" i know (the answer|it) now",
]
# Questions about the bot's question: "which project are you asking about?"
_CLARIFY = [
    r" (which|what) (project|one|model|part|kind|type|company|example|aspect|area|role)s? (are|do|did|should|should i|you)",
    r" (are|were) you asking (about|for)", r" what do you mean", r" do you mean", r" (be )?more specific",
    r" should i (talk|explain|tell|speak|answer|focus|describe)", r" (any|which) (particular|specific) (project|one|example)",
    r" what (project|model|work) did i mention", r"^ which (project|one) ", r" you mean (which|the)",
    r" (can|could) you (clarify|explain the question|be specific)", r" what do you want me to",
    r" what (exactly )?should i (say|explain|walk|talk|tell|cover)", r" give me (some )?clarity",
    r" what exactly (do|are|did) you", r" walk you through what",
]
CLARIFY_MAX = 25  # a question about the question can run longer than SHORT
# "Hello, can you hear me?"
_HEAR = [
    r" (can|could) you (hear|listen to) me", r" are you (there|listening|hearing me)", r" is (this|it) working",
    r" am i audible", r"^ hello( hello)*( are you there)? $",
]
# "Can you repeat what I said?"
_READ_BACK = [
    r" (repeat|tell me|say) what i (said|just said|told)", r" what did i (just )?(say|tell)",
    r" did you (get|hear|catch) (it|that|me|what i said)",
]
_SKIP = [
    r" (what'?s|what is) (my|the) next question", r" next question", r" (go|move) (on )?to the next",
    r"^ (next|move on|skip)( please)? $", r" skip (this|it|the question)", r"^ skip ", r"^ pass $", r" i'?ll pass ",
    r" let'?s move on",
]
_REFUSE = [
    r" (don'?t|do not) (want|wish|like) to (answer|speak|talk|reply|say)", r" what if i (don'?t|do not) (want to )?answer",
    r" i (won'?t|refuse to|will not) (answer|speak|talk)", r" i'?m not (going to|gonna) answer",
]
_END = [
    r" (stop|end|quit|leave|finish) (the|this) interview", r" i want to (stop|quit|leave|end)( now)? $",
    r" i (don'?t|do not) want to (continue|do this interview)",
]
# Complaints about the bot itself.
_COMPLAINT = [
    r" (annoying|irritating|frustrating)", r" (stop|don'?t keep|quit) (saying|repeating)",
    r" why (are|aren'?t) you( not)? (stopping|listening)", r" you (keep|are) (repeating|interrupting|saying)",
    r" keep switching", r" stop interrupting",
]
_DONT_KNOW = [
    r" i don'?t know", r" i do not know", r" no idea", r" i'?m not aware", r" not aware of",
    r" i don'?t remember", r" i haven'?t (worked|used|done|tried)", r" never (worked|used) ",
    r" i can'?t answer", r" no clue", r" dono ", r" dunno", r" don'?t no ",
]
_NEED_TIME = [
    r" give me (a|one) (minute|moment|second|sec|min) ", r" let me think", r"^ (one|a|just a) (minute|moment|second|sec) ",
    r" hold on ", r" can i (think|take a (minute|moment))", r" need (a|some) (time|moment|minute) ",
    r" just a (moment|second|sec) ", r"^ wait ", r" (good|interesting|tough|tricky|nice) question",
    r"^ (hmm |umm |um )?let me (see|recall) ",
]
_HINT = [r" hint", r" (a|any) clue ", r" can you help me", r" give me an example", r" help me (out|with) "]
_SALARY = [
    r" (salary|ctc|compensation|stipend|hike|notice period)", r" what (is|about) the package",
    r" package (offered|details)",
]
_IDENTITY = [
    r" are you (a |an )?(real|human|bot|robot|ai|person|machine|recording)", r" am i talking to",
    r" is this (a |an )?(bot|ai|recording|real person|human)",
]
_PROCESS = [r" how many questions", r" how long (is|will)", r" how much time", r" when will i (know|hear)"]
_DOUBT = [
    r" i have (a|one|small|a small) (doubt|question|dot)", r" (can|may|could) i ask", r" quick question",
    r"^ (one )?doubt ",
]

_AFFIRM = [
    r"^ (yes|yeah|yep|yup|sure|ok|okay|ready|absolutely|definitely|haan|ha|ji|of course|go ahead|please) ",
    r" i'?m ready", r" i am ready", r" let'?s (go|start|begin|do (it|this)) ", r" yes ",
    r"^ (start|begin)( now)? ",
]
_NEGATE = [r"^ (no|nope|not|nah|nahi) ", r" not (yet|ready|now) ", r" i'?m (good|fine|ok) ", r" no thanks"]


def is_short(text: str) -> bool:
    return len(_words(text)) <= SHORT


def detect(text: str) -> str | None:
    """Return an intent for short special replies, or None for a normal answer.

    Possible values: previous, repeat, dont_know, need_time, hint_request,
    salary, identity, process, doubt.
    """
    if not is_short(text):
        n = len(_words(text))
        if n <= CLARIFY_MAX and _has(text, _CLARIFY) and not _has(text, _PREVIOUS):
            return "clarify_question"
        return None
    # Order matters: "repeat the previous question" is a revisit, and
    # "sorry, can you repeat? I don't know what you mean" is a repeat.
    if _has(text, _COMPLAINT):
        return "complaint"
    if _has(text, _END):
        return "end_request"
    if _has(text, _REFUSE):
        return "refuse"
    if _has(text, _HEAR):
        return "hear_check"
    if _has(text, _READ_BACK):
        return "read_back"
    if _has(text, _GO_ON):
        return "go_on"
    if _has(text, _PREVIOUS):
        return "previous"
    if _has(text, _CLARIFY):
        return "clarify_question"
    if _has(text, _REPEAT):
        return "repeat"
    if _has(text, _IDENTITY):
        return "identity"
    if _has(text, _SALARY):
        return "salary"
    if _has(text, _PROCESS):
        return "process"
    if _has(text, _HINT):
        return "hint_request"
    if _has(text, _NEED_TIME):
        return "need_time"
    if _has(text, _SKIP):
        return "skip"
    if _has(text, _DOUBT):
        return "doubt"
    # "I'm not sure but I think..." is the start of an answer, not a give-up.
    if _has(text, _DONT_KNOW) and " but " not in f" {normalize(text)} ":
        return "dont_know"
    return None


_OUTSIDE_HELP = re.compile(
    r"\b(chat ?gpt|claude|gemini|copilot|perplexity|bard|google|the internet|ai tool)\b[^.?!]{0,40}"
    r"\b(told|tells|telling|said|says|gave|gives|answer|answered|helped|helping)\b"
    r"|\b(someone|somebody|my friend|a friend|my colleague)\b[^.?!]{0,20}\b(told|telling|helping|helped|whispering)\b"
    r"|\bi(?:'m| am) (reading|looking) (it )?(from|at|up|it)\b"
    r"|\b(googl(e|ed|ing) it|looked it up|search(ed|ing) (it|online))\b",
    re.I,
)


def mentions_outside_help(text: str) -> bool:
    """'Claude told me the right answer' (said twice in the TL's test)."""
    return bool(_OUTSIDE_HELP.search(text))


def is_affirmative(text: str) -> bool:
    return is_short(text) and _has(text, _AFFIRM) and not _has(text, _NEGATE)


def is_negative(text: str) -> bool:
    return is_short(text) and _has(text, _NEGATE)
