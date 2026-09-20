from typing import List, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from retrieval.memory import formulate_query
from retrieval.search import retrieve_context
from generation.llm import generate_answer

# define state (shared memory)
class RAGState(TypedDict):
    query: str
    chat_history: List[Dict[str, str]]
    standalone_query: str
    context: str
    answer: str

# define nodes (agents)
def query_agent(state: RAGState) -> dict:
    import app
    new_query=formulate_query(state["query"], state["chat_history"], app.llm_client, app.LLM_MODEL)
    return {"standalone_query":new_query}

def retrieval_agent(state: RAGState) -> dict:
    import app
    docs=retrieve_context(state["standalone_query"], app.embed_model, app.collection, app.reranker_model)
    return {"context":docs}

def generation_agent(state: RAGState) -> dict:
    import app
    ans=generate_answer(state["query"], state["context"], state["chat_history"], app.llm_client, app.LLM_MODEL)
    return {"answer":ans}


# build graph

workflow=StateGraph(RAGState)

workflow.add_node("router",query_agent)
workflow.add_node("retriever",retrieval_agent)
workflow.add_node("generator",generation_agent)

workflow.add_edge(START, "router")
workflow.add_edge("router", "retriever")
workflow.add_edge("retriever","generator")
workflow.add_edge("generator",END)

agentic_rag=workflow.compile()