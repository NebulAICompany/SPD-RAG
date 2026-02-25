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
    model="gemini-2.5-pro",
    temperature=0.0,
    callbacks=[ConsoleCallbackHandler()],
)

RESEARCH_LLM_REASONING = GEMINI_25_FLASH
RESEARCH_LLM_FAST = GEMINI_25_FLASH


SELECTED_FILES: Optional[List[str]] = None

ORIGINAL_USER_QUERY: Optional[str] = None


def set_selected_files(files: Optional[List[str]]) -> None:
    """Set the global selected files for RAG queries."""
    global SELECTED_FILES
    SELECTED_FILES = files


def get_selected_files() -> Optional[List[str]]:
    """Get the global selected files for RAG queries."""
    return SELECTED_FILES


def set_original_user_query(query: Optional[str]) -> None:
    """Set the global original user query."""
    global ORIGINAL_USER_QUERY
    ORIGINAL_USER_QUERY = query


def get_original_user_query() -> Optional[str]:
    """Get the global original user query."""
    return ORIGINAL_USER_QUERY
