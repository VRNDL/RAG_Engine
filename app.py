from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import json
import uuid
import redis.asyncio as redis
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv
from workflow import agentic_rag
from prometheus_fastapi_instrumentator import Instrumentator

# global variables
collection = None
embed_model = None
reranker_model = None
llm_client = None
redis_client = None 
LLM_MODEL = "openai/gpt-oss-120b"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection, embed_model, reranker_model, llm_client, redis_client
    load_dotenv()

    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY")
    )
    
    # Connect to Docker Redis
    redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    yield
    await redis_client.close()

app = FastAPI(lifespan=lifespan)

# Expose FastAPI metrics to Prometheus on the /metrics endpoint
Instrumentator().instrument(app).expose(app)

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] = []
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    chat_history: List[Dict[str, str]]
    session_id: str
    strategy_used: str
    error: Optional[str] = None

class FeedbackRequest(BaseModel):
    strategy_used: str
    is_positive: bool

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
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

    final_state = agentic_rag.invoke(initial_state, config=config)
    
    answer = final_state.get("answer", "")
    error = final_state.get("error")

    updated_history = request.chat_history.copy()
    if not error:
        updated_history.append({"role": "user", "content": request.query})
        updated_history.append({"role": "assistant", "content": answer})

        if len(updated_history) > 6:
            updated_history = updated_history[-6:]

    return ChatResponse(
        answer=answer if not error else "Error generating response.",
        chat_history=updated_history,
        session_id=thread_id,
        strategy_used=final_state.get("strategy", "unknown"),
        error=error
    )

@app.post("/feedback")
def feedback_endpoint(request: FeedbackRequest):
    from router_logic import log_feedback
    try:
        log_feedback(request.strategy_used, request.is_positive)
        return {"status": "success", "message": "Feedback integrated into Thompson Sampling"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_ip = websocket.client.host
    
    try:
        while True:
            # 1. Wait for frontend to send a message
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            # 2. Redis Rate Limiting (Max 5 requests per 10 seconds per IP)
            requests = await redis_client.incr(f"rate_limit:{client_ip}")
            if requests == 1:
                await redis_client.expire(f"rate_limit:{client_ip}", 10)
            if requests > 5:
                await websocket.send_text(json.dumps({"error": "Rate limit exceeded. Please slow down."}))
                continue

            # 3. Process via LangGraph
            thread_id = payload.get("session_id", str(uuid.uuid4()))
            config = {"configurable": {"thread_id": thread_id}}
            
            initial_state = {
                "query": payload["query"],
                "chat_history": payload.get("chat_history", []),
                "standalone_query": "",
                "context": "",
                "answer": "",
                "error": None
            }

            # Tell the frontend we are thinking
            await websocket.send_text(json.dumps({"status": "processing", "agent": "router"}))
            
            # Execute graph
            final_state = agentic_rag.invoke(initial_state, config=config)
            
            # 4. Return result over WebSocket
            response = {
                "answer": final_state.get("answer", "Error"),
                "session_id": thread_id,
                "strategy_used": final_state.get("strategy", "unknown"),
                "error": final_state.get("error")
            }
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        print(f"Client {client_ip} disconnected")