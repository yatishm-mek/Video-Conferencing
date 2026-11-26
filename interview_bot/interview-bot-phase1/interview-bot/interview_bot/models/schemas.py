"""Data contracts shared across phases (pydantic).

Phase 2 uses QuestionPlan. EvalResult (Phase 4) and TurnLog (Phase 6) are
added when those phases are built.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Topic = Literal["intro", "project", "jd_skill", "fundamentals", "experience", "motivation"]


class Question(BaseModel):
    id: int
    topic: Topic
    q: str = Field(description="The question as the bot should ask it")
    expected: list[str] = Field(default_factory=list, description="Rubric points")
    max_followups: int = 2
    context: str = Field(
        default="",
        description="What the question is about: the resume or JD item it targets, by name (no numbers)",
    )


class QuestionPlan(BaseModel):
    candidate: str
    role: str = ""
    level: str = Field(default="", description='Experience the JD asks for, e.g. "0-1 years"')
    vocab: list[str] = Field(default_factory=list, description="Tech terms from resume + JD")
    questions: list[Question]

    def _name_parts(self) -> list[str]:
        # Skip initials: "M. Sai Shashank" -> ["Sai", "Shashank"]
        return [p for p in self.candidate.split() if len(p.strip(".")) > 1]

    @property
    def first_name(self) -> str:
        parts = self._name_parts()
        return parts[0] if parts else ""

    def stt_vocab(self) -> list[str]:
        """Terms to bias Whisper: candidate's name first, then tech terms."""
        names = self._name_parts()
        return names + [v for v in self.vocab if v not in names]

    @classmethod
    def load(cls, path: str | Path) -> QuestionPlan:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


# -- Evaluator contract (the controller consumes it from Phase 3; Phase 4 fills
#    it with the LLM evaluator, Phase 3 with the rule-based stub) -------------

Intent = Literal[
    "answer",
    "clarify_question",
    "earlier_question",
    "repeat",
    "dont_know",
    "need_time",
    "hint_request",
    "off_topic",
    "injection_attempt",
    "other_question",
]


class Correction(BaseModel):
    from_: str = Field(alias="from")
    to: str

    model_config = {"populate_by_name": True}


class UnclearTerm(BaseModel):
    heard: str
    best_guess: str
    importance: Literal["low", "medium", "high"]


class EvalResult(BaseModel):
    turn_id: str
    raw_transcript: str
    interpreted_answer: str
    corrections: list[Correction] = Field(default_factory=list)
    unclear_terms: list[UnclearTerm] = Field(default_factory=list)
    intent: Intent = "answer"
    complete: bool = True
    verdict: Literal["correct", "partial", "vague", "suspicious", "unanswered"] = "partial"
    score: int = Field(default=0, ge=0, le=10)
    rubric_hits: list[str] = Field(default_factory=list)
    notes: str = ""
    flags: list[str] = Field(default_factory=list)
    action: Literal["next", "follow_up", "verify", "clarify"] = "next"
    follow_up_question: str | None = None

    @classmethod
    def default_next(cls, turn_id: str, transcript: str, note: str = "") -> EvalResult:
        """What the controller uses when the evaluator fails or times out."""
        return cls(
            turn_id=turn_id,
            raw_transcript=transcript,
            interpreted_answer=transcript,
            action="next",
            notes=note,
        )
