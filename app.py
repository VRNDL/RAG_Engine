from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv
from workflow import agentic_rag
import uuid

# global variables
collection = None  # Kept as None so workflow.py doesn't crash when importing app.collection
embed_model = None
reranker_model = None
llm_client = None
LLM_MODEL = "openai/gpt-oss-120b"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs ONCE when server starts
    global collection, embed_model, reranker_model, llm_client
    load_dotenv()

    # ChromaDB initialization removed - Weaviate handles retrieval in search.py now
    
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY")
    )
    yield

# start-up
app = FastAPI(lifespan=lifespan)

# definition of JSON structure
class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] = []
    session_id: Optional[str] = None  # Added for PostgreSQL tracking

# just class definition
class ChatResponse(BaseModel):
    answer: str
    chat_history: List[Dict[str, str]]
    session_id: str
    stratergy_used: str
    error: Optional[str] = None

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Generate a unique thread ID if the client didn't provide one
    thread_id = request.session_id if request.session_id else str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "query": request.query,
        "chat_history": request.chat_history,
        "standalone_query": "",
        "context": "",
        "answer": "",
        "error": None
    }

    # Execute graph with Postgres persistence
    final_state = agentic_rag.invoke(initial_state, config=config)
    
    answer = final_state.get("answer", "")
    error = final_state.get("error")

    # append to history (only if no error occurred)
    updated_history = request.chat_history.copy()
    if not error:
        updated_history.append({"role": "user", "content": request.query})
        updated_history.append({"role": "assistant", "content": answer})

        if len(updated_history) > 6:
            updated_history = updated_history[-6:]

    # return answer, history, and the session ID to the client
    return ChatResponse(
        answer=answer if not error else "Error generating response.",
        chat_history=updated_history,
        session_id=thread_id,
        strategy_used=final_state.get("strategy", "unknown"),
        error=error
    )

class FeedbackRequest(BaseModel):
    strategy_used: str
    is_positive: bool

@app.post("/feedback")
def feedback_endpoint(request: FeedbackRequest):
    from router_logic import log_feedback
    try:
        log_feedback(request.strategy_used, request.is_positive)
        return {"status": "success", "message": "Feedback integrated into Thompson Sampling"}
    except Exception as e:
        return {"status": "error", "message": str(e)}