"""SPD-RAG FastAPI backend: upload, chat over the compiled LangGraph, and session state.

Exposes /upload (document indexing), /chat (query with selected_documents), and
/health, /files. The graph runs coordination -> parallel sub-agents -> synthesis.
"""

import sys
import os

rrm_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if rrm_path not in sys.path:
    sys.path.append(rrm_path)

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import shutil
from backend.pipeline.upload import process_file
from backend.shared.constants import UPLOADS_PATH_STR

from backend.core.graph import get_compiled_graph
from backend.shared.logger import get_logger
from backend.retrieval.retriever import load_vectorstore, close_vectorstore
from backend.shared.constants import (
    VECTORSTORE_PATH_STR,
    set_selected_files,
    set_original_user_query,
)
from langchain_core.messages import HumanMessage, AIMessage

logger = get_logger("SPD_RAG_API")


class QueryRequest(BaseModel):
    """Chat request: user query, optional session id, optional document filter."""

    query: str
    session_id: Optional[str] = "default_session"
    selected_files: Optional[List[str]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load vectorstore and compile graph on startup; close vectorstore on shutdown."""
    logger.info("Starting SPD-RAG API")
    if os.path.exists(VECTORSTORE_PATH_STR):
        logger.info("Loading vectorstore...")
        load_vectorstore(VECTORSTORE_PATH_STR)
    else:
        logger.warning("Vectorstore not found at %s", VECTORSTORE_PATH_STR)

    app.state.graph = get_compiled_graph()

    yield

    logger.info("Shutting down SPD-RAG API")
    close_vectorstore()


app = FastAPI(title="SPD-RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory chat history per session; replace with a database for production
CHAT_HISTORY: Dict[str, List[Any]] = {}


@app.get("/health")
async def health_check():
    """Liveness check for the API."""
    return {"status": "healthy"}


@app.get("/files")
async def get_files():
    """Return list of uploaded file names from the uploads directory."""
    try:
        if not os.path.exists(UPLOADS_PATH_STR):
            return {"files": []}
            
        files = [f for f in os.listdir(UPLOADS_PATH_STR) if os.path.isfile(os.path.join(UPLOADS_PATH_STR, f))]
        return {"files": files}
    except Exception as e:
        logger.error(f"Error getting files: {e}")
        return {"files": []}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Save the uploaded file, then parse, chunk, embed, and index it for retrieval."""
    try:
        os.makedirs(UPLOADS_PATH_STR, exist_ok=True)
        file_path = os.path.join(UPLOADS_PATH_STR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info("File saved to %s", file_path)

        result = await process_file(file_path)
        
        if result["status"] == "error":
             raise HTTPException(status_code=500, detail=result["message"])
             
        return result
        
    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    """Run the SPD-RAG graph: coordination -> sub-agents -> synthesis; return final answer."""
    try:
        query = request.query
        session_id = request.session_id
        selected_files = request.selected_files

        logger.info("Chat query (session=%s): %s", session_id, query[:80] if query else "")

        set_selected_files(selected_files)
        set_original_user_query(query)

        if session_id not in CHAT_HISTORY:
            CHAT_HISTORY[session_id] = []

        history = CHAT_HISTORY[session_id]
        messages = list(history)
        messages.append(HumanMessage(content=query))

        graph = app.state.graph
        config = {"configurable": {"thread_id": session_id}}

        result = await graph.ainvoke(
            {
                "messages": messages,
                "selected_documents": selected_files or [],
            },
            config=config,
        )

        answer = "I couldn't generate a response."
        if result.get("messages"):
            for msg in reversed(result["messages"]):
                if msg.content:
                    answer = msg.content
                    break

        CHAT_HISTORY[session_id].append(HumanMessage(content=query))
        CHAT_HISTORY[session_id].append(AIMessage(content=answer))

        return {
            "response": answer,
            "session_id": session_id
        }

    except Exception as e:
        logger.error("Chat endpoint error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8001, reload=True)
