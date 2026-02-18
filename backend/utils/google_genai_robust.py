import asyncio
import re
import logging
from typing import Any, AsyncIterator, Iterator, List, Union

from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
    wait_exponential,
    before_sleep_log,
    RetryCallState,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from google.api_core.exceptions import (
    ResourceExhausted,
    ServiceUnavailable,
    InternalServerError,
)

logger = logging.getLogger(__name__)

_RETRYABLE_EXCEPTIONS = (
    ResourceExhausted,
    ServiceUnavailable,
    InternalServerError,
)


def _extract_wait_time(e: Exception, attempt: int) -> float:
    if isinstance(e, ResourceExhausted):
        match = re.search(
            r"retry_delay.*?seconds:\s*(\d+)", str(e), re.IGNORECASE | re.DOTALL
        )
        if match:
            return float(int(match.group(1)) + 1)
    return min(2**attempt, 60)


def _google_smart_wait(retry_state: RetryCallState) -> float:
    exception = retry_state.outcome.exception()
    if exception and isinstance(exception, ResourceExhausted):
        match = re.search(
            r"retry_delay.*?seconds:\s*(\d+)",
            str(exception),
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            wait_time = int(match.group(1)) + 1
            logger.warning(f"Quota hit! Google requested wait: {wait_time}s")
            return float(wait_time)
    return wait_exponential(multiplier=1, min=2, max=60)(retry_state)


def _build_retry():
    return retry(
        stop=stop_after_attempt(10),
        wait=_google_smart_wait,
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


class RobustChatGoogleGenerativeAI:
    def __init__(self, model: str, **kwargs):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            max_retries=1,
            **kwargs,
        )

    def invoke(self, input: Union[str, List[BaseMessage]], **kwargs) -> Any:
        return _build_retry()(self.llm.invoke)(input, **kwargs)

    async def ainvoke(self, input: Union[str, List[BaseMessage]], **kwargs) -> Any:
        return await _build_retry()(self.llm.ainvoke)(input, **kwargs)

    def stream(self, input: Union[str, List[BaseMessage]], **kwargs) -> Iterator:
        return _build_retry()(self.llm.stream)(input, **kwargs)

    async def astream(
        self, input: Union[str, List[BaseMessage]], **kwargs
    ) -> AsyncIterator:
        for attempt in range(10):
            try:
                async for chunk in self.llm.astream(input, **kwargs):
                    yield chunk
                return
            except _RETRYABLE_EXCEPTIONS as e:
                if attempt == 9:
                    raise
                wait_time = _extract_wait_time(e, attempt)
                logger.warning(
                    f"astream retry {attempt + 1}/10, waiting {wait_time:.1f}s — {e}"
                )
                await asyncio.sleep(wait_time)

    def bind_tools(self, tools, **kwargs) -> "RobustChatGoogleGenerativeAI":
        new = RobustChatGoogleGenerativeAI.__new__(RobustChatGoogleGenerativeAI)
        new.llm = self.llm.bind_tools(tools, **kwargs)
        return new

    def with_structured_output(
        self, schema, **kwargs
    ) -> "RobustChatGoogleGenerativeAI":
        new = RobustChatGoogleGenerativeAI.__new__(RobustChatGoogleGenerativeAI)
        new.llm = self.llm.with_structured_output(schema, **kwargs)
        return new

    def bind(self, **kwargs) -> "RobustChatGoogleGenerativeAI":
        new = RobustChatGoogleGenerativeAI.__new__(RobustChatGoogleGenerativeAI)
        new.llm = self.llm.bind(**kwargs)
        return new

    def __getattr__(self, name: str) -> Any:
        if name == "llm":
            raise AttributeError("Inner LLM not initialized")
        return getattr(self.llm, name)
