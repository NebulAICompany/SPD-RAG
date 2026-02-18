import re
import logging
from typing import Any, List, Union, Optional

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


def google_smart_wait(retry_state: RetryCallState) -> float:
    """
    Custom wait strategy that parses 'retry_delay' from Google's 429 error.
    Falls back to exponential backoff for other errors.
    """
    exception = retry_state.outcome.exception()

    if exception and isinstance(exception, ResourceExhausted):
        error_str = str(exception)
        match = re.search(
            r"retry_delay.*?seconds:\s*(\d+)", error_str, re.IGNORECASE | re.DOTALL
        )
        if match:
            wait_time = int(match.group(1)) + 1
            logger.warning(f"Quota hit! Google requested wait: {wait_time}s")
            return float(wait_time)

    return wait_exponential(multiplier=1, min=2, max=60)(retry_state)


class RobustChatGoogleGenerativeAI:
    """
    Production-ready wrapper for ChatGoogleGenerativeAI.
    Fixes the known issue where SDK ignores server-side retry suggestions.
    """

    def __init__(self, model: str, **kwargs):
        kwargs.pop("max_retries", None)
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            max_retries=1,
            **kwargs,
        )

    def _create_retry_decorator(self):
        return retry(
            stop=stop_after_attempt(10),
            wait=google_smart_wait,
            retry=retry_if_exception_type(
                (
                    ResourceExhausted,
                    ServiceUnavailable,
                    InternalServerError,
                )
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    async def ainvoke(self, input: Union[str, List[BaseMessage]], **kwargs) -> Any:
        decorator = self._create_retry_decorator()
        return await decorator(self.llm.ainvoke)(input, **kwargs)

    def invoke(self, input: Union[str, List[BaseMessage]], **kwargs) -> Any:
        decorator = self._create_retry_decorator()
        return decorator(self.llm.invoke)(input, **kwargs)
