# robust_google_chat.py
import asyncio
import os
import re
import time
import threading
import logging
from typing import Any, AsyncIterator, Iterator, List, Union

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from google.genai.errors import ServerError, ClientError

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 15

# 5xx codes worth retrying (transient server-side issues)
_RETRYABLE_SERVER_CODES = {500, 502, 503, 504}
# Only 429 is key-retryable for client errors; 400/401/403/404 should propagate immediately
_QUOTA_CLIENT_CODES = {429}


# ── Error classification ──────────────────────────────────────────────────────

def _is_retryable(e: Exception) -> bool:
    """True only for transient server errors and quota/rate-limit (429) responses."""
    if isinstance(e, ServerError):
        return getattr(e, "code", 0) in _RETRYABLE_SERVER_CODES
    if isinstance(e, ClientError):
        return getattr(e, "code", 0) in _QUOTA_CLIENT_CODES
    return False


def _is_quota_error(e: Exception) -> bool:
    """True when rotating to a different API key may help (quota exhausted on this key)."""
    return isinstance(e, ClientError) and getattr(e, "code", 0) == 429


def _extract_wait_time(e: Exception, attempt: int) -> float:
    """
    Return seconds to sleep before the next retry.
    Priority: structured retryDelay in e.details → string regex fallbacks → exponential default.
    """
    # 1. Structured retryDelay (most accurate — present on 429 responses)
    try:
        details = e.details.get("error", {}).get("details", [])  # type: ignore[attr-defined]
        for d in details:
            m = re.match(r"^(\d+)s$", d.get("retryDelay", ""))
            if m:
                return float(m.group(1)) + 1.0
    except Exception:
        pass

    error_str = str(e)

    # 2. 503 overload → aggressive 3^n backoff (server capacity issue)
    if isinstance(e, ServerError) and "503" in error_str:
        return min(3 ** attempt, 90)

    # 3. String-encoded retry hints (older SDK versions)
    m = re.search(r"retry_delay.*?seconds:\s*(\d+)", error_str, re.IGNORECASE | re.DOTALL)
    if m:
        return float(m.group(1)) + 1.0

    m = re.search(r'"retryDelay":\s*"(\d+)s"', error_str, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 1.0

    # 4. Default: capped exponential backoff
    return min(2 ** attempt, 60)


# ── Key loading ───────────────────────────────────────────────────────────────

def _load_api_keys() -> List[str]:
    """
    Load API keys from .env.
    Reads GOOGLE_API_KEYS (comma-separated) first; falls back to GOOGLE_API_KEY.
    """
    load_dotenv()
    keys = [k.strip() for k in os.getenv("GOOGLE_API_KEYS", "").split(",") if k.strip()]
    if not keys:
        single = os.getenv("GOOGLE_API_KEY", "").strip()
        if single:
            keys = [single]
    return keys


# ── Main class ────────────────────────────────────────────────────────────────

class RobustChatGoogleGenerativeAI:
    """
    A resilient ChatGoogleGenerativeAI wrapper with:
    - Smart per-error wait-time extraction (structured → regex → exponential)
    - Retry only on genuinely transient errors (5xx, 429); hard-fail on 4xx
    - Round-robin API key rotation on quota exhaustion (429)
    - Keys loaded from GOOGLE_API_KEYS (comma-separated) or GOOGLE_API_KEY in .env
    - bind_tools / bind / with_structured_output applied across all key instances

    .env format:
        GOOGLE_API_KEYS=AIzaSy...key1,AIzaSy...key2,AIzaSy...key3
    """

    # ── Constructors ──────────────────────────────────────────────────────────

    def __init__(self, model: str, **kwargs):
        api_keys = _load_api_keys()
        if not api_keys:
            raise ValueError(
                "No API keys found. Set GOOGLE_API_KEYS (comma-separated) "
                "or GOOGLE_API_KEY in your .env file."
            )
        self._key_index = 0
        self._lock = threading.Lock()
        self._llms: List[Any] = [
            ChatGoogleGenerativeAI(model=model, api_key=key, max_retries=1, **kwargs)
            for key in api_keys
        ]
        logger.info("Initialized with %d API key(s).", len(self._llms))

    @classmethod
    def _from_llms(cls, llms: List[Any]) -> "RobustChatGoogleGenerativeAI":
        """Internal factory used by bind_tools / bind / with_structured_output."""
        obj = cls.__new__(cls)
        obj._llms = llms
        obj._key_index = 0
        obj._lock = threading.Lock()
        return obj

    # ── Key management ────────────────────────────────────────────────────────

    @property
    def _current_llm(self) -> Any:
        return self._llms[self._key_index]

    def _rotate_key(self) -> None:
        if len(self._llms) <= 1:
            return
        with self._lock:
            old = self._key_index
            self._key_index = (self._key_index + 1) % len(self._llms)
        logger.warning("API key rotated: index %d → %d", old, self._key_index)

    # ── Shared retry helper ───────────────────────────────────────────────────

    def _wait_or_raise(self, e: Exception, attempt: int, label: str) -> float:
        """
        Called from within an except block.
        Re-raises (bare `raise`) if the error is non-retryable or max attempts reached;
        otherwise rotates the key if needed and returns the wait duration in seconds.
        """
        if not _is_retryable(e) or attempt >= MAX_ATTEMPTS - 1:
            raise  # bare raise preserves the original traceback
        if _is_quota_error(e):
            self._rotate_key()
        wait = _extract_wait_time(e, attempt)
        logger.warning(
            "%s retry %d/%d, waiting %.1fs — %s: %s",
            label, attempt + 1, MAX_ATTEMPTS, wait,
            type(e).__name__, e,
        )
        return wait

    # ── Public API ────────────────────────────────────────────────────────────

    def invoke(self, input: Union[str, List[BaseMessage]], **kwargs) -> Any:
        for attempt in range(MAX_ATTEMPTS):
            try:
                return self._current_llm.invoke(input, **kwargs)
            except Exception as e:
                time.sleep(self._wait_or_raise(e, attempt, "invoke"))

    async def ainvoke(self, input: Union[str, List[BaseMessage]], **kwargs) -> Any:
        for attempt in range(MAX_ATTEMPTS):
            try:
                return await self._current_llm.ainvoke(input, **kwargs)
            except Exception as e:
                await asyncio.sleep(self._wait_or_raise(e, attempt, "ainvoke"))

    def stream(self, input: Union[str, List[BaseMessage]], **kwargs) -> Iterator:
        # NOTE: must be a generator (yield from) to catch mid-stream errors.
        # Wrapping with tenacity only retries the *initial call*, not the iteration.
        for attempt in range(MAX_ATTEMPTS):
            try:
                yield from self._current_llm.stream(input, **kwargs)
                return
            except Exception as e:
                time.sleep(self._wait_or_raise(e, attempt, "stream"))

    async def astream(self, input: Union[str, List[BaseMessage]], **kwargs) -> AsyncIterator:
        for attempt in range(MAX_ATTEMPTS):
            try:
                async for chunk in self._current_llm.astream(input, **kwargs):
                    yield chunk
                return
            except Exception as e:
                await asyncio.sleep(self._wait_or_raise(e, attempt, "astream"))

    # ── Chaining helpers ──────────────────────────────────────────────────────
    # Apply to every LLM instance so key rotation is preserved after binding.

    def bind_tools(self, tools, **kwargs) -> "RobustChatGoogleGenerativeAI":
        return self._from_llms([llm.bind_tools(tools, **kwargs) for llm in self._llms])

    def with_structured_output(self, schema, **kwargs) -> "RobustChatGoogleGenerativeAI":
        return self._from_llms([llm.with_structured_output(schema, **kwargs) for llm in self._llms])

    def bind(self, **kwargs) -> "RobustChatGoogleGenerativeAI":
        return self._from_llms([llm.bind(**kwargs) for llm in self._llms])

    def __getattr__(self, name: str) -> Any:
        # Guard against infinite recursion for uninitialized private attrs
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        return getattr(self._current_llm, name)
