import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv

from retrieval.search import retrieve_context
from retrieval.memory import formulate_query
from generation.llm import generate_answer

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rag_engine_pipeline"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "openai/gpt-oss-120b"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def main():
    load_dotenv()
    
    print("Loading database and models...")
    db_client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = db_client.get_collection(name=COLLECTION_NAME)
    
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
    
    llm_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    
    # Initialize an empty list to store our short-term memory
    chat_history = []
    
    print("\n==================================================")
    print("Conversational RAG Online. Type 'quit' to exit.")
    print("==================================================\n")
    
    while True:
        user_query = input("User: ")
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
            
        # 1. Rewrite the query if it relies on past context
        standalone_query = formulate_query(user_query, chat_history, llm_client, LLM_MODEL)
        if standalone_query != user_query:
            print(f"   [Contextualized Search]: {standalone_query}")
            
        # 2. Search ChromaDB using the STANDALONE query
        context = retrieve_context(standalone_query, embed_model, collection, reranker_model)
        
        # 3. Generate the answer using the ORIGINAL query and chat history
        answer = generate_answer(user_query, context, chat_history, llm_client, LLM_MODEL)
        print(f"\nAssistant: {answer}\n")
        
        # 4. Save to Memory (keep only last 6 messages to avoid bloating the prompt)
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": answer})
        if len(chat_history) > 6:
            chat_history = chat_history[-6:]

if __name__ == "__main__":
    main()