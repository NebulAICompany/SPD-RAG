import os
from typing import List, Optional
from dotenv import load_dotenv
import cohere
from pathlib import Path
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_openai import ChatOpenAI

load_dotenv()

# Constants for API keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Constants for API clients
co = cohere.ClientV2(api_key=COHERE_API_KEY)

# Constants for file paths
BASE_DIR = Path(__file__).parent.parent  # backend/ directory
PROJECT_ROOT = BASE_DIR.parent  # AIris/ or RRM/ directory

# Database paths
DATABASE_DIR = BASE_DIR / "database"

# Upload and document paths
UPLOADS_PATH = DATABASE_DIR / "uploads"

# Vectorstore paths
VECTORSTORE_PATH = DATABASE_DIR / "vectorstore"

# Logging paths
LOGS_DIR = PROJECT_ROOT / "logs"
BACKEND_LOG_PATH = LOGS_DIR / "backend.log"
BACKEND_ERROR_LOG_PATH = LOGS_DIR / "backend_errors.log"

# Convert Path objects to strings for backward compatibility
UPLOADS_PATH_STR = str(UPLOADS_PATH)
VECTORSTORE_PATH_STR = str(VECTORSTORE_PATH)

BACKEND_LOG_PATH_STR = str(BACKEND_LOG_PATH)
BACKEND_ERROR_LOG_PATH_STR = str(BACKEND_ERROR_LOG_PATH)

# Research Models
RESEARCH_LLM_REASONING = ChatOpenAI(
    model="gpt-5", # Updated to a valid model name as gpt-5 is likely not available or valid yet
    temperature=0.0,
    callbacks=[ConsoleCallbackHandler()],
)
RESEARCH_LLM_FAST = ChatOpenAI(
    model="gpt-5-mini", # Updating to standard fast model
    temperature=0.0,
    callbacks=[ConsoleCallbackHandler()],
)

# Global variable for selected files in RAG queries
SELECTED_FILES: Optional[List[str]] = None

# Global variable for original user query
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

