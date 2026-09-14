import sys
from sentence_transformers import SentenceTransformer
from ingestion.pipeline import load_document, chunk_text, generate_embeddings,store_chunks

PERSIST_DIR="./chroma_db"
COLLECTION_NAME="rag_engine_pipeline"
EMBEDDING_MODEL_NAME="BAAI/bge-small-en-v1.5"

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_ingest.py <path_to_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    print("[1/4] Loading document...")
    raw_text = load_document(file_path)
    
    print("[2/4] Splitting text...")
    chunks = chunk_text(raw_text, source=file_path)
    
    print("[3/4] Embedding...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = generate_embeddings(chunks, embed_model)
    
    print("[4/4] Storing in ChromaDB...")
    total = store_chunks(chunks, embeddings, PERSIST_DIR, COLLECTION_NAME)
    print(f"Done! Collection now holds {total} records.")

if __name__ == "__main__":
    main()