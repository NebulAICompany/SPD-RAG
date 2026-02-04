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

from backend.agent_graph.graph import get_compiled_graph
from backend.shared.logger import get_logger
from backend.retrieval.retriever import load_vectorstore, close_vectorstore
from backend.shared.constants import (
    VECTORSTORE_PATH_STR,
    set_selected_files,
    set_original_user_query,
)
from langchain_core.messages import HumanMessage, AIMessage

logger = get_logger("RRM_API")

# Request Models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default_session"
    selected_files: Optional[List[str]] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting RRM API...")
    if os.path.exists(VECTORSTORE_PATH_STR):
        logger.info("Loading vectorstore...")
        load_vectorstore(VECTORSTORE_PATH_STR)
    else:
        logger.warning(f"Vectorstore not found at {VECTORSTORE_PATH_STR}")
    
    # Initialize graph
    app.state.graph = get_compiled_graph()
    
    yield
    
    # Shutdown
    logger.info("Shutting down RRM API...")
    close_vectorstore()

app = FastAPI(title="RRM Agent API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory chat history for simple session management
# In production, use a database (like the original implementations chat_history_manager)
CHAT_HISTORY: Dict[str, List[Any]] = {}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/files")
async def get_files():
    """Get list of uploaded files"""
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
    """Upload and process a file for RAG"""
    try:
        # Save file to uploads directory
        os.makedirs(UPLOADS_PATH_STR, exist_ok=True)
        file_path = os.path.join(UPLOADS_PATH_STR, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"File saved to {file_path}")
        
        # Process the file
        result = await process_file(file_path)
        
        if result["status"] == "error":
             raise HTTPException(status_code=500, detail=result["message"])
             
        return result
        
    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        query = request.query
        session_id = request.session_id
        selected_files = request.selected_files
        
        logger.info(f"Received query: {query} (Session: {session_id})")
        
        # 1. Set global context for RAG tools
        set_selected_files(selected_files)
        set_original_user_query(query)
        
        # 2. Manage Chat History
        if session_id not in CHAT_HISTORY:
            CHAT_HISTORY[session_id] = []
            
        history = CHAT_HISTORY[session_id]
        
        # Prepare messages
        messages = [msg for msg in history] # Copy existing
        messages.append(HumanMessage(content=query))
        
        # 3. Invoke Graph
        graph = app.state.graph
        config = {"configurable": {"thread_id": session_id}}
        
        result = await graph.ainvoke(
            {
                "messages": messages,
                "selected_documents": selected_files or [],
            },
            config=config
        )
        
        # 4. Extract Answer
        answer = "I couldn't generate a response."
        if result.get("messages"):
            for msg in reversed(result["messages"]):
                if msg.content:
                    answer = msg.content
                    break
        
        # 5. Update History
        CHAT_HISTORY[session_id].append(HumanMessage(content=query))
        CHAT_HISTORY[session_id].append(AIMessage(content=answer))
        
        # 6. Return Response
        return {
            "response": answer,
            "session_id": session_id
        }

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
