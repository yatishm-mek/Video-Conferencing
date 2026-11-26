"""Phase 1 stand-in for the LLM: repeats the candidate's last turn.

It sits exactly where the LLM service will go in Phase 2 and speaks the same
frame protocol (LLMContextFrame in -> LLMFullResponseStart/Text/End out), so
swapping it for OllamaLLMService changes one line of pipeline.py.
"""

from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# Say "long test" to get a long reply you can interrupt mid-sentence.
LONG_REPLY = (
    "Sure, here is a long reply so you can practise interrupting me. "
    "In this interview I will ask you about your projects, the skills listed in the job "
    "description, and a few fundamentals. Take your time with each answer. "
    "If you want me to repeat a question, just ask. "
    "Try talking over me now, and I should stop within about half a second. "
    "If you only say okay or hmm, I should keep going."
)


def last_user_text(frame: LLMContextFrame) -> str:
    for msg in reversed(frame.context.get_messages()):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):  # content parts
                return " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                ).strip()
    return ""


def echo_reply(user_text: str) -> str:
    if "long test" in user_text.lower():
        return LONG_REPLY
    return f"You said: {user_text}" if user_text else "Sorry, I didn't catch that."


class EchoResponder(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            text = last_user_text(frame)
            reply = echo_reply(text)
            logger.info(f"ECHO user={text!r} -> bot={reply[:60]!r}")
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame(reply))
            await self.push_frame(LLMFullResponseEndFrame())
            return
        await self.push_frame(frame, direction)
