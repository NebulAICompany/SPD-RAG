"""SPD-RAG configuration: paths, API clients (Cohere, OpenAI, Gemini), and global query state.

Cohere is used for embeddings (embed-v4.0) and reranking (rerank-v4.0-fast). Gemini
models are wrapped with RobustChatGoogleGenerativeAI for retries and key rotation.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv
import cohere
from pathlib import Path
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_openai import ChatOpenAI
from backend.utils.google_genai_robust import RobustChatGoogleGenerativeAI

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

co = cohere.ClientV2(api_key=COHERE_API_KEY)

BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BASE_DIR.parent

DATABASE_DIR = BASE_DIR / "database"

UPLOADS_PATH = DATABASE_DIR / "uploads"

VECTORSTORE_PATH = DATABASE_DIR / "vectorstore"

LOGS_DIR = PROJECT_ROOT / "logs"
BACKEND_LOG_PATH = LOGS_DIR / "backend.log"
BACKEND_ERROR_LOG_PATH = LOGS_DIR / "backend_errors.log"

UPLOADS_PATH_STR = str(UPLOADS_PATH)
VECTORSTORE_PATH_STR = str(VECTORSTORE_PATH)

BACKEND_LOG_PATH_STR = str(BACKEND_LOG_PATH)
BACKEND_ERROR_LOG_PATH_STR = str(BACKEND_ERROR_LOG_PATH)

GPT5 = ChatOpenAI(
    model="gpt-5",
    temperature=0.0,
    callbacks=[ConsoleCallbackHandler()],
)
GPT5_MINI = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0.0,
    callbacks=[ConsoleCallbackHandler()],
)

GEMINI_25_FLASH = RobustChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    callbacks=[ConsoleCallbackHandler()],
)
GEMINI_25_PRO = RobustChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.0,
    callbacks=[ConsoleCallbackHandler()],
)

RESEARCH_LLM_REASONING = GEMINI_25_PRO
RESEARCH_LLM_FAST = GEMINI_25_FLASH


SELECTED_FILES: Optional[List[str]] = None

ORIGINAL_USER_QUERY: Optional[str] = None


def set_selected_files(files: Optional[List[str]]) -> None:
    """Set the list of document names to restrict retrieval (used by RAG tools)."""
    global SELECTED_FILES
    SELECTED_FILES = files


def get_selected_files() -> Optional[List[str]]:
    """Return the current selected-documents filter, or None for no filter."""
    return SELECTED_FILES


def set_original_user_query(query: Optional[str]) -> None:
    """Set the root user query (e.g. for synthesis or tool context)."""
    global ORIGINAL_USER_QUERY
    ORIGINAL_USER_QUERY = query


def get_original_user_query() -> Optional[str]:
    """Return the current root user query, or None."""
    return ORIGINAL_USER_QUERY


def get_synthesizer_token_limit_for_fast() -> int:
    """Return a safe per-batch token limit for the synthesis layer (e.g. 750k for Gemini).

    Derived from the configured RESEARCH_LLM_FAST model context size.
    """
    max_context_map = {
        "gpt-5": 400_000,
        "gpt-5-mini": 400_000,
        "gemini-2.5-flash": 1_000_000,
        "gemini-2.5-pro": 1_000_000,
        "gemini-3-flash": 200_000,
        "gemini-3-pro": 1_000_000,
    }

    model_name = getattr(RESEARCH_LLM_FAST, "model_name", None) or getattr(
        RESEARCH_LLM_FAST, "model", None
    )

    max_ctx = max_context_map.get(model_name, 128_000)
    return int(max_ctx * 0.75)
