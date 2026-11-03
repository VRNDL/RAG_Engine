from typing import List, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from retrieval.memory import formulate_query
from retrieval.search import retrieve_context
from generation.llm import generate_answer

# 1. Define state (added 'error' for deterministic control)
class RAGState(TypedDict):
    query: str
    chat_history: List[Dict[str, str]]
    standalone_query: str
    context: str
    answer: str
    error: str 

# 2. Define nodes with Supervisor Error Catching
def query_agent(state: RAGState) -> dict:
    import app
    try:
        new_query = formulate_query(state["query"], state["chat_history"], app.llm_client, app.LLM_MODEL)
        return {"standalone_query": new_query, "error": None}
    except Exception as e:
        return {"error": f"Query Agent Failed: {str(e)}"}

def retrieval_agent(state: RAGState) -> dict:
    import app
    try:
        docs = retrieve_context(state["standalone_query"], app.embed_model, app.collection, app.reranker_model)
        return {"context": docs, "error": None}
    except Exception as e:
        return {"error": f"Retrieval Agent Failed: {str(e)}"}

def generation_agent(state: RAGState) -> dict:
    import app
    try:
        ans = generate_answer(state["query"], state["context"], state["chat_history"], app.llm_client, app.LLM_MODEL)
        return {"answer": ans, "error": None}
    except Exception as e:
        return {"error": f"Generation Agent Failed: {str(e)}"}

def error_recovery_agent(state: RAGState) -> dict:
    # Deterministic fallback if any agent throws an exception
    print(f"-> [Supervisor] Handling Error: {state.get('error')}")
    return {"answer": "I encountered an internal system error while processing that. Please try again."}

# 3. Define Conditional Routing Logic
def route_query(state: RAGState):
    return "error_recovery" if state.get("error") else "retriever"

def route_retrieval(state: RAGState):
    return "error_recovery" if state.get("error") else "generator"

def route_generation(state: RAGState):
    return "error_recovery" if state.get("error") else END

# 4. Build graph
workflow = StateGraph(RAGState)

workflow.add_node("router", query_agent)
workflow.add_node("retriever", retrieval_agent)
workflow.add_node("generator", generation_agent)
workflow.add_node("error_recovery", error_recovery_agent)

workflow.add_edge(START, "router")
# Replaced linear edges with conditional routing
workflow.add_conditional_edges("router", route_query)
workflow.add_conditional_edges("retriever", route_retrieval)
workflow.add_conditional_edges("generator", route_generation)
workflow.add_edge("error_recovery", END)

# 5. Compile with PostgreSQL State Persistence
DB_URI = "postgresql://rag_user:rag_password@localhost:5433/rag_evals"

# Added autocommit=True to allow LangGraph to create indexes concurrently
pool = ConnectionPool(
    conninfo=DB_URI,
    kwargs={"autocommit": True}
)
checkpointer = PostgresSaver(pool)

# This automatically creates the tracking tables in your Docker Postgres container
checkpointer.setup() 

agentic_rag = workflow.compile(checkpointer=checkpointer)