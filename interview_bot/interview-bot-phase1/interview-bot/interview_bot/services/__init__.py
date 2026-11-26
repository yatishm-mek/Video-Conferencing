"""External / model-backed services the live loop depends on.

Each module here wraps one capability behind a small surface:
  llm.py          - Ollama chat client (shared HTTP pool)
  stt.py          - speech-to-text factory (local faster-whisper | groq) + filters
  tts.py          - text-to-speech (Kokoro) + line cache
  bot_speaker.py  - the interviewer LLM (asks / rephrases the questions)
  evaluator.py    - the evaluator LLM (scores each answer -> EvalResult)
"""
