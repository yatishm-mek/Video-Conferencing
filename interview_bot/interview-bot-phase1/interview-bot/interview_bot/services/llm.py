"""Minimal async client for Ollama's native API (/api/chat, /api/ps).

We use the native endpoint, not Ollama's OpenAI-compatible /v1 that Pipecat's
OLLamaLLMService wraps, because it lets us:
  * turn thinking off per call (`think: false`) - the live loop must not think,
  * set the context window per call (`options.num_ctx`), so the model fits the
    GPU without relying on OLLAMA_CONTEXT_LENGTH being set on the machine,
  * keep the model loaded (`keep_alive`) and read GPU placement (/api/ps).

Works the same against a CPU-only Ollama, which is the deployment target.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import aiohttp
from loguru import logger

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _as_keep_alive(value: str | int | float) -> str | int | float:
    """Normalise keep_alive for Ollama's native API.

    Ollama accepts either a number of seconds (-1 = keep the model loaded
    forever, 0 = unload immediately) or a duration string *with a unit*
    ("30m", "24h"). A bare numeric string like "-1" is not a valid Go duration,
    so /api/chat rejects it with 400 'time: missing unit in duration "-1"' and
    the model never loads. Send numeric strings as numbers; leave unit'd
    strings untouched. ("0" happens to be accepted either way.)
    """
    if isinstance(value, (int, float)):
        return value
    s = str(value).strip()
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


@dataclass
class ChatStats:
    first_token_secs: float | None = None
    total_secs: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    load_secs: float = 0.0
    prompt_eval_secs: float = 0.0
    extra: dict = field(default_factory=dict)


class OllamaClient:
    def __init__(self, base_url: str, *, num_ctx: int = 4096, keep_alive: str = "30m"):
        self._base = base_url.rstrip("/")
        self._num_ctx = num_ctx
        self._keep_alive = _as_keep_alive(keep_alive)
        self._session: aiohttp.ClientSession | None = None

    async def _http(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=5)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        *,
        think: bool = False,
        temperature: float = 0.4,
        max_tokens: int | None = None,
        stats: ChatStats | None = None,
        format: dict | str | None = None,
    ) -> AsyncIterator[str]:
        """Yield the reply's text as it streams. Fills `stats` when done."""
        options: dict = {"num_ctx": self._num_ctx, "temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": think,
            "options": options,
            "keep_alive": self._keep_alive,
        }
        if format is not None:
            body["format"] = format  # JSON schema: Ollama constrains output to it
        t0 = time.monotonic()
        http = await self._http()
        async with http.post(f"{self._base}/api/chat", json=body) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Ollama /api/chat {resp.status}: {(await resp.text())[:300]}")
            in_think = False
            async for raw in resp.content:
                line = raw.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise RuntimeError(f"Ollama error: {chunk['error']}")
                text = chunk.get("message", {}).get("content", "")
                # Defensive: some model templates still inline <think> blocks.
                if "<think>" in text:
                    in_think = True
                if in_think:
                    if "</think>" in text:
                        in_think = False
                        text = text.split("</think>", 1)[1]
                    else:
                        text = ""
                if text:
                    if stats and stats.first_token_secs is None:
                        stats.first_token_secs = time.monotonic() - t0
                    yield text
                if chunk.get("done"):
                    if stats:
                        stats.total_secs = time.monotonic() - t0
                        stats.prompt_tokens = chunk.get("prompt_eval_count", 0)
                        stats.output_tokens = chunk.get("eval_count", 0)
                        stats.load_secs = chunk.get("load_duration", 0) / 1e9
                        stats.prompt_eval_secs = chunk.get("prompt_eval_duration", 0) / 1e9
                    load = chunk.get("load_duration", 0) / 1e9
                    if load > 1.0:
                        logger.warning(f"[llm] Ollama (re)loaded {model} for this call: {load:.1f}s")
                    break

    async def chat(self, model: str, messages: list[dict], **kwargs) -> str:
        parts = [t async for t in self.chat_stream(model, messages, **kwargs)]
        return _THINK_BLOCK.sub("", "".join(parts)).strip()

    async def loaded_models(self) -> list[dict]:
        http = await self._http()
        async with http.get(f"{self._base}/api/ps") as resp:
            resp.raise_for_status()
            return (await resp.json()).get("models", [])

    async def warmup(self, model: str, prime: list[dict] | None = None) -> None:
        """Load the model, run requests shaped like the real ones, report GPU placement.

        A tiny "Say OK." is not enough: on the RTX 5050 the first *real*
        ~300-token prompt still took 2.3 s to read (later ones 0.22 s), because
        a longer prompt takes a different, still-cold GPU path. Priming with the
        live system prompt also leaves it in Ollama's prompt cache.
        """
        t0 = time.monotonic()
        await self.chat(
            model, [{"role": "user", "content": "Say OK."}], max_tokens=2, temperature=0.0
        )
        if prime:
            await self.chat(model, prime, max_tokens=4, temperature=0.0)
        logger.info(f"LLM {model} warmed up in {time.monotonic() - t0:.1f}s")
        for m in await self.loaded_models():
            if m.get("name") == model or m.get("model") == model:
                size, vram = m.get("size", 0), m.get("size_vram", 0)
                pct = 100 * vram / size if size else 0
                ctx = m.get("context_length", "?")
                where = (
                    "100% GPU"
                    if pct >= 99.5
                    else ("CPU" if pct == 0 else f"{100 - pct:.0f}% CPU / {pct:.0f}% GPU")
                )
                msg = f"LLM {model}: {where}, {size / 1e9:.1f} GB, context {ctx}"
                if 0 < pct < 99.5:
                    logger.warning(
                        msg + " - part of the model is on the CPU, so replies will be slow. "
                        "Use a smaller LIVE_LLM_MODEL (e.g. qwen3.5:4b) or a smaller OLLAMA_NUM_CTX."
                    )
                else:
                    logger.info(msg)


_CLIENT: OllamaClient | None = None


def get_llm(settings) -> OllamaClient:
    """One client (one HTTP connection pool) for the whole server."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OllamaClient(
            settings.OLLAMA_BASE_URL,
            num_ctx=settings.OLLAMA_NUM_CTX,
            keep_alive=settings.OLLAMA_KEEP_ALIVE,
        )
    return _CLIENT
