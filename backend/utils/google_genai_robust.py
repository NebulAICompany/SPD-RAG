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
from google.genai.errors import ServerError, ClientError

logger = logging.getLogger(__name__)

_RETRYABLE_EXCEPTIONS = (ServerError, ClientError)


def _extract_wait_time(e: Exception, attempt: int) -> float:
    error_str = str(e)

    if isinstance(e, ServerError) and "503" in error_str:
        wait = min(3**attempt, 90)
        logger.warning(f"503 detected, aggressive backoff: {wait:.1f}s")
        return wait

    match = re.search(
        r"retry_delay.*?seconds:\s*(\d+)", error_str, re.IGNORECASE | re.DOTALL
    )
    if match:
        return float(int(match.group(1)) + 1)

    match = re.search(r'"retryDelay":\s*"(\d+)s"', error_str, re.IGNORECASE)
    if match:
        return float(int(match.group(1)) + 1)

    return min(2**attempt, 60)


def _google_smart_wait(retry_state: RetryCallState) -> float:
    exception = retry_state.outcome.exception()
    if exception:
        return _extract_wait_time(exception, retry_state.attempt_number - 1)
    return 1.0


def _build_retry():
    return retry(
        stop=stop_after_attempt(15),
        wait=_google_smart_wait,
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


class RobustChatGoogleGenerativeAI:
    def __init__(self, model: str, **kwargs):
        self.llm = ChatGoogleGenerativeAI(model=model, max_retries=1, **kwargs)

    def invoke(self, input: Union[str, List[BaseMessage]], **kwargs) -> Any:
        return _build_retry()(self.llm.invoke)(input, **kwargs)

    async def ainvoke(self, input: Union[str, List[BaseMessage]], **kwargs) -> Any:
        return await _build_retry()(self.llm.ainvoke)(input, **kwargs)

    def stream(self, input: Union[str, List[BaseMessage]], **kwargs) -> Iterator:
        return _build_retry()(self.llm.stream)(input, **kwargs)

    async def astream(
        self, input: Union[str, List[BaseMessage]], **kwargs
    ) -> AsyncIterator:
        for attempt in range(15):
            try:
                async for chunk in self.llm.astream(input, **kwargs):
                    yield chunk
                return
            except _RETRYABLE_EXCEPTIONS as e:
                if attempt == 14:
                    raise
                wait_time = _extract_wait_time(e, attempt)
                logger.warning(
                    f"astream retry {attempt + 1}/15, waiting {wait_time:.1f}s"
                )
                await asyncio.sleep(wait_time)

    def bind_tools(self, tools, **kwargs):
        new = RobustChatGoogleGenerativeAI.__new__(RobustChatGoogleGenerativeAI)
        new.llm = self.llm.bind_tools(tools, **kwargs)
        return new

    def with_structured_output(self, schema, **kwargs):
        new = RobustChatGoogleGenerativeAI.__new__(RobustChatGoogleGenerativeAI)
        new.llm = self.llm.with_structured_output(schema, **kwargs)
        return new

    def bind(self, **kwargs):
        new = RobustChatGoogleGenerativeAI.__new__(RobustChatGoogleGenerativeAI)
        new.llm = self.llm.bind(**kwargs)
        return new

    def __getattr__(self, name: str) -> Any:
        if name == "llm":
            raise AttributeError("Inner LLM not initialized")
        return getattr(self.llm, name)
