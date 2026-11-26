"""All thresholds, model names and switches. Loaded from .env (see .env.example).

Every timing rule in the design lives here so it can be tuned without code changes.
Later phases read the same object; nothing else should hardcode a threshold.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- Server -------------------------------------------------------------
    HOST: str = "127.0.0.1"
    PORT: int = 7860

    # ---- Storage (nothing is written outside these two folders) -------------
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    DATA_DIR: Path = PROJECT_ROOT / "data"

    # ---- VAD (Silero) -------------------------------------------------------
    VAD_CONFIDENCE: float = 0.7
    VAD_START_SECS: float = 0.2
    VAD_STOP_SECS: float = 0.2
    VAD_MIN_VOLUME: float = 0.6

    # ---- Turn-taking / barge-in ---------------------------------------------
    # Candidate must speak this long while the bot is talking before TTS is cut.
    # Shorter bursts ("okay", "hmm", "mm-hmm") are treated as backchannels.
    BARGE_IN_MIN_SECS: float = 0.5
    # Speech longer than this while the bot talks cuts it without waiting for
    # the words. Between MIN and HARD, the transcript decides (backchannel or not).
    # Sustained speech this long cuts the bot without waiting for words. 1.0 s let
    # speaker echo cut nearly every question in the TL's test.
    # 1.2 s: talking over the bot stops it in about 1.4 s. If speaker echo cuts the bot
    # twice, the echo guard moves this to 2.0 s for the rest of the call. Headphones help.
    BARGE_IN_HARD_SECS: float = 1.2
    # Words that alone mean "I'm listening", "|"-separated. Empty = built-in list.
    BACKCHANNEL_WORDS: str = ""
    # If the candidate speaks within this window after the bot starts talking,
    # cut TTS immediately: the bot jumped in before their answer was done.
    EARLY_BARGE_IN_WINDOW_SECS: float = 0.3
    # Safety net: end an open user turn after this much inactivity.
    USER_TURN_STOP_TIMEOUT_SECS: float = 5.0

    # ---- STT -----------------------------------------------------------------
    # local  = faster-whisper in-process (Pipecat WhisperSTTService)
    # groq   = Groq cloud Whisper
    # moonshine = the fine-tuned Moonshine Vosk-WS server (custom-asr), reused as a
    #             one-shot client so the bot can listen with the same model Jitsi uses.
    STT_PROVIDER: Literal["local", "groq", "moonshine"] = "local"
    # Where the bot's Moonshine STT connects. Use the ASR ROUTER on the Jitsi box
    # (:2700), not the Moonshine backend directly (:2710): the router is the exposed
    # port (firewall opens :2700), defaults to Moonshine, and proxies the Vosk
    # protocol to it. Override with MOONSHINE_WS_URL in the env/.env.
    MOONSHINE_WS_URL: str = "ws://192.168.7.174:2700"
    STT_LANGUAGE: str = "en"
    # faster-whisper (CTranslate2) build of large-v3-turbo
    WHISPER_MODEL: str = "deepdml/faster-whisper-large-v3-turbo-ct2"
    WHISPER_DEVICE: str = "auto"  # auto | cpu | cuda
    WHISPER_COMPUTE_TYPE: str = "int8"  # int8 on CPU; int8_float16 is faster on CUDA
    WHISPER_BEAM_SIZE: int = 1  # faster-whisper defaults to 5; 1 is much faster
    # Segments whose no_speech_prob is >= this are dropped.
    WHISPER_NO_SPEECH_PROB: float = 0.4
    # Segments that compress better than this are Whisper looping ("BG, BG, BG...").
    WHISPER_MAX_COMPRESSION_RATIO: float = 2.4
    # Segments Whisper is this unsure of are dropped (noise, echo, other languages:
    # "Repito, ai sera nao", "Awal Nia Chait, Putaner" in the TL's test).
    WHISPER_MIN_AVG_LOGPROB: float = -1.0
    # Max wait for the final transcript after end-of-turn (Pipecat default 1.0).
    STT_TTFS_P99_SECS: float = 1.5
    GROQ_API_KEY: str = ""
    GROQ_STT_MODEL: str = "whisper-large-v3-turbo"
    # Phase 1 only: stand-in for QuestionPlan.vocab (comma-separated).
    STT_VOCAB: str = "LoRA, PyTorch, CTC, WER, Whisper, fine-tuning, transformer, BM25"
    # Whole-segment Whisper hallucinations to drop, "|"-separated.
    HALLUCINATION_PHRASES: str = (
        "thank you for watching|thanks for watching|please subscribe|"
        "subscribe to my channel|like and subscribe|subscribe|you|thank you|amen"
    )

    # ---- TTS -----------------------------------------------------------------
    # qwen   = Qwen3-TTS via the microservice: "base" (the 1.7B base model, replaces
    #          Kokoro as the default voice) or a cloned voice. Slower on the current GPU.
    # kokoro = legacy Kokoro-82M (kept but no longer the default).
    TTS_PROVIDER: Literal["qwen", "kokoro"] = "qwen"
    KOKORO_VOICE: str = "af_heart"
    KOKORO_SPEED: float = 1.0
    # Qwen3-TTS microservice — see integration/tts_service.
    QWEN_TTS_URL: str = "http://127.0.0.1:8099"
    QWEN_VOICE: str = "base"       # base | rithika | aditi | sharanya | suresh
    QWEN_STYLE: str = "balanced"   # steady | balanced | expressive

    # ---- LLM (Ollama, native API) ----------------------------------------------
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"  # not "localhost": ~2 s per new connection on Windows
    # 4096 keeps qwen3.5:9b 100% on an 8 GB GPU (16K spilled 16% to CPU).
    OLLAMA_NUM_CTX: int = 4096
    # -1 = keep the live model loaded. With 30m it was unloaded after an idle
    # afternoon, and the next interview's first replies timed out while it reloaded.
    OLLAMA_KEEP_ALIVE: str = "-1"
    # Live loop (bot speaker, later the evaluator): thinking OFF, latency critical.
    # Same model for both; only one LLM is ever loaded in the live loop.
    LIVE_LLM_MODEL: str = "qwen3.5:9b"
    LIVE_LLM_TEMPERATURE: float = 0.4
    LIVE_LLM_MAX_TOKENS: int = 80
    # No first words within this -> the bot asks the question verbatim instead.
    LLM_FIRST_TOKEN_TIMEOUT_SECS: float = 4.0
    # Evaluator: llm = Qwen judges each answer (JSON); rules = accept everything.
    EVALUATOR: Literal["llm", "rules"] = "llm"
    EVAL_TEMPERATURE: float = 0.1
    EVAL_MAX_TOKENS: int = 260
    # Prep (question plan) and report: thinking ON, not latency critical.
    PREP_LLM_MODEL: str = "qwen3.5:9b"
    PREP_NUM_CTX: int = 8192  # resume + JD + thinking
    PREP_THINK: bool = False  # on: thinking used up all 6000 tokens on 8K context (live test)
    PREP_NUM_QUESTIONS: int = 6
    PREP_MAX_TOKENS: int = 6000

    # ---- Interview -------------------------------------------------------------
    # interview = Qwen asks the plan's questions; echo = Phase 1 audio test bot.
    BOT_MODE: Literal["interview", "echo"] = "interview"
    # natural: one LLM turn per reply (the model talks like an interviewer, code keeps
    # the hard rules). classic: the 4-prompt state machine with fixed replies.
    BOT_ENGINE: Literal["natural", "classic"] = "natural"
    NATURAL_TEMPERATURE: float = 0.5
    NATURAL_MAX_TOKENS: int = 150
    NATURAL_RESCUE_SECS: float = 1.8  # no reply audio by then: one short "Mm, okay."
    NATURAL_RESCUE_EVERY_TURNS: int = 3  # ...at most once every 3 turns
    NATURAL_TURN_TIMEOUT_SECS: float = 5.0
    LABEL_FORMAT: str = "json"  # two_step: how the model reports what the candidate meant: json | compact (faster on CPU)
    NATURAL_MODE: str = "llm"  # two_step: model labels + code decides + model words the reply | llm: one call does all | rules: older code-rule engine
    # Give the candidate real thinking time before the bot nudges with "would you like a
    # moment?". 5 s was too eager (it nagged while they were still composing an answer).
    NATURAL_SILENCE_PROMPT_SECS: float = 12.0
    NATURAL_SILENCE_MOVE_ON_SECS: float = 25.0
    MAX_REPEATS_PER_QUESTION: int = 3  # user's rule: 3 per question (2nd: "listen carefully", 3rd: "last time")
    MAX_GO_BACKS_PER_INTERVIEW: int = 1
    MAX_IDLE_TURNS_PER_QUESTION: int = 4
    MAX_END_CHECKS_PER_INTERVIEW: int = 2  # "should I end?" asked this many times; the next time the bot ends it  # replies with no answer content before only moving on is allowed
    SCORE_MAX_TOKENS: int = 300
    # Phase 2: a hardcoded plan. Phase 5 generates it from resume + JD.
    INTERVIEW_PLAN_PATH: Path = PROJECT_ROOT / "data" / "sample_plan.json"

    # ---- Controller timers (used from Phase 3) --------------------------------
    NO_SPEECH_PROMPT_SECS: float = 5
    INCOMPLETE_PROMPT_SECS: float = 5
    EXTENDED_WAIT_SECS: float = 25
    # Filler ("Okay, got it.") while the evaluator works; evals take 2-3 s, so speak early.
    EVALUATOR_FILLER_SECS: float = 0.8
    # Spec said 3; measured qwen3.5:9b evals take 2.4-3.7 s, so 3 dropped most follow-ups.
    EVALUATOR_TIMEOUT_SECS: float = 5
    MAX_ANSWER_SECS: float = 180
    MAX_CONFIRMATIONS_PER_INTERVIEW: int = 3
    MAX_VERIFICATIONS_PER_INTERVIEW: int = 3
    EVAL_HISTORY_QUESTIONS: int = 4
    EVAL_HISTORY_WORDS: int = 60
    INTERVIEW_TIME_LIMIT_SECS: float = 20 * 60
    # No reply to "Would you like a moment?" for this long -> extended wait.
    NEED_TIME_REPLY_SECS: float = 6
    # Greeting asked "Are you ready?" and got silence -> one gentle reprompt.
    READY_REPROMPT_SECS: float = 12
    # After an answer of HOLD_MIN_WORDS+ words, wait this much silence before
    # moving on, so a thinking pause doesn't get the next question. 0 = off.
    ANSWER_HOLD_SECS: float = 1.0
    HOLD_MIN_WORDS: int = 12
    # Clarification limits: 1st "didn't get it" -> verbatim, 2nd -> reworded
    # (at most MAX_REPHRASES per interview), Nth -> "give it your best attempt"
    # + flag. FLAG_REPEATS_TOTAL requests across the interview -> flag.
    MAX_REPHRASES_PER_INTERVIEW: int = 3
    REPEATS_BEFORE_BEST_ATTEMPT: int = 3
    FLAG_REPEATS_TOTAL: int = 5
    # Asking to go back to an earlier question is declined; flagged from this many.
    FLAG_GO_BACK_REQUESTS: int = 2
    # Turn ended after this much silence = Smart Turn gave up waiting (its own
    # limit is ~3 s), so the candidate may have trailed off -> "Is that your complete answer?"
    DONE_CHECK_SILENCE_SECS: float = 2.3
    # Cutting in after this share of a new question = early answer to it.
    EARLY_ANSWER_FRACTION: float = 0.6
    # Rejoining within this many seconds continues the same interview.
    RESUME_MAX_AGE_SECS: float = 900
    # Bot audio must stay stopped this long to count as "finished speaking".
    BOT_FINISHED_DEBOUNCE_SECS: float = 0.8
    # Speech was sent but no audio started within this -> carry on anyway.
    SPEECH_WATCHDOG_SECS: float = 8

    # ---- Diagnostics -----------------------------------------------------------
    LOG_LEVEL: str = "DEBUG"
    # Log incoming mic loudness every N seconds (0 = off).
    AUDIO_LEVEL_LOG_SECS: float = 2.0
    # Opt-in deep tracing (interview_bot.debug): off keeps behaviour unchanged.
    # On adds a rotating logs/debug.log, [trace] lines for controller actions and
    # turns, and JSON snapshots under logs/debug/ for after-the-fact inspection.
    DEBUG_TRACE: bool = False

    @field_validator("INTERVIEW_PLAN_PATH", "LOGS_DIR", "DATA_DIR")
    @classmethod
    def _relative_to_project(cls, v: Path) -> Path:
        # .env paths like "data/sample_plan.json" mean the project folder,
        # whichever folder the server was started from.
        return v if v.is_absolute() else PROJECT_ROOT / v

    @property
    def hallucination_phrases(self) -> list[str]:
        return [p.strip() for p in self.HALLUCINATION_PHRASES.split("|") if p.strip()]

    @property
    def stt_vocab_list(self) -> list[str]:
        return [t.strip() for t in self.STT_VOCAB.split(",") if t.strip()]


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    s.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return s
