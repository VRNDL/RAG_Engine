import sys
import os
from sentence_transformers import SentenceTransformer
from ingestion.pipeline import load_document, chunk_text, generate_embeddings,store_chunks
from openai import OpenAI
from dotenv import load_dotenv
from ingestion.contextualizer import contextualize_chunks


PERSIST_DIR="./chroma_db"
COLLECTION_NAME="rag_engine_pipeline"
EMBEDDING_MODEL_NAME="BAAI/bge-small-en-v1.5"
CONTEXTUAL_LLM_MODEL="openai/gpt-oss-120b"


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_ingest.py <path_to_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    load_dotenv()
    
    print("[1/5] Loading document...")
    raw_text = load_document(file_path)
    
    print("[2/5] Splitting text...")
    chunks = chunk_text(raw_text, source=file_path)
    
    print("[3/5] Contextualizing chunks (Anthropic Method)...")
    llm_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    chunks = contextualize_chunks(chunks, raw_text, llm_client, CONTEXTUAL_LLM_MODEL)
    
    print("[4/5] Embedding...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = generate_embeddings(chunks, embed_model)
    
    print("[5/5] Storing in ChromaDB...")
    total = store_chunks(chunks, embeddings, PERSIST_DIR, COLLECTION_NAME)
    print(f"Done! Collection now holds {total} records.")

if __name__ == "__main__":
    main()