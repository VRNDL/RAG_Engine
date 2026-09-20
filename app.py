from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv

from retrieval.search import retrieve_context
from retrieval.memory import formulate_query
from generation.llm import generate_answer

# global variables, else lifespan will treat it as local variables
db_client=None
collection=None
embed_model=None
reranker_model=None
llm_client=None
LLM_MODEL="openai/gpt-oss-120b"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs ONCE when server starts
    global db_client, collection, embed_model, reranker_model, llm_client
    load_dotenv()

    db_client=chromadb.PersistentClient(path="./chroma_db")
    collection=db_client.get_collection(name="rag_engine_pipeline")
    embed_model=SentenceTransformer("BAAI/bge-small-en-v1.5")
    reranker_model=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm_client=OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY")
    )
    yield

# start-up
app=FastAPI(lifespan=lifespan)

# definition of JSON structure
class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]]=[]

# just class definition
class ChatResponse(BaseModel):
    answer: str
    chat_history: List[Dict[str, str]]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # reformulate
        standalone_query=formulate_query(request.query, request.chat_history, llm_client, LLM_MODEL)

        # retrieve
        context=retrieve_context(standalone_query, embed_model, collection, reranker_model)

        # generate
        answer=generate_answer(request.query, context, request.chat_history, llm_client, LLM_MODEL)

        # append to history
        updated_history=request.chat_history.copy()
        updated_history.append({"role":"user", "content":request.query})
        updated_history.append({"role":"assistant", "content":answer})

        if(len(updated_history)>6):
            updated_history=updated_history[-6:]

        # return answer and history to client
        return ChatResponse(answer=answer, chat_history=updated_history)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
