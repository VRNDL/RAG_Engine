import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv

from retrieval.search import retrieve_context
from generation.llm import generate_answer

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rag_engine_pipeline"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "openai/gpt-oss-120b"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def main():
    if len(sys.argv) < 2:
        print('Usage: python run_query.py "Your question"')
        sys.exit(1)
        
    user_query = sys.argv[1]
    load_dotenv()
    
    print("[1/3] Loading database and models...")
    db_client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = db_client.get_collection(name=COLLECTION_NAME)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    reranker_model=CrossEncoder(RERANKER_MODEL_NAME)
    llm_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    
    print("[2/3] Searching database...")
    context = retrieve_context(user_query, embed_model, collection, reranker_model)

    #print(f"\n--- RAW RETRIEVED CHUNKS ---\n{context}\n----------------------------\n")
    
    print("[3/3] Generating answer...")
    answer = generate_answer(user_query, context, llm_client, LLM_MODEL)
    
    print(f"\n--- QUESTION: {user_query} ---\n")
    print(answer)
    print("\n--------------------------------")

if __name__ == "__main__":
    main()