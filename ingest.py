import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter  #chunking
from sentence_transformers import SentenceTransformer  #embedding
import chromadb   # vector database

#--Config--

PERSIST_DIR="./chroma_db"                      # where to save
COLLECTION_NAME="rag_engine_baseline"
EMBEDDING_MODEL_NAME="BAAI/bge-small-en-v1.5"
CHUNK_SIZE=500                                 # characters per chunk
CHUNK_OVERLAP=50   


#--Loader--

def load_document(file_path: str) -> str:
    path=Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    ext=path.suffix.lower()   # get extension

    if ext==".pdf":
        reader=PdfReader(str(path))     #open pdf
        # extract text page by page
        pages_text=[page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages_text)                # form one big string of content

    elif ext in [".txt",".md"]:
        with open(path,"r",encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported format. Expected: .pdf, .md, .txt")


#--Chunking--

def chunk_text(raw_text: str, source: str) -> List[Dict[str, Any]]:
    splitter=RecursiveCharacterTextSplitter(   #init
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n","\n"," ",""]  # what to split on (priority order)
    )

    splits=splitter.split_text(raw_text)   # list of strings after splitting
    chunks=[]

    for idx, text in enumerate(splits):
        # put each chunk in a dict
        chunks.append({                              # putting dicts in a list
            "id" : f"{Path(source).stem}_{idx}",     # file name and index as key
            "text" : text,
            "metadata" : {
                "source": str(source),
                "chunk_index":idx,
                "char_length":len(text)
            }
        })
    return chunks


#--Embedding--

def generate_embeddings(chunks: List[Dict[str, Any]], model: SentenceTransformer):
    texts=[c["text"] for c in chunks]  # extract raw text from the dicts
    # scaled for cosine similarity
    embeddings=model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.tolist()  # numpy to list


#--Storage of chunks--

def store_chunks(chunks: List[Dict[str, Any]], embeddings: List[List[float]], persist_dir: str):
    client=chromadb.PersistentClient(path=persist_dir)  # connect to chromadb 
    collection=client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # what to search by
    )

    # break chunks into separate lists for chromadb
    ids=[c["id"] for c in chunks]
    documents=[c["text"] for c in chunks]   
    metadatas=[c["metadata"] for c in chunks]

    # update database
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    return collection.count()

def verify_retrieval(test_query: str, persist_dir: str, model: SentenceTransformer, n_results: int = 3):
    """Sanity check: embed a test query, search the store, print results."""
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    query_embedding = model.encode([test_query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    print(f"\n--- Test query: '{test_query}' ---")
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        print(f"  [{dist:.4f}] {doc[:150]}...")

#--Orchestration--

def main():
    # Check if the user forgot to pass a file path in the terminal
    # sys.argv is the list of command line arguments. Index 0 is the script name, Index 1 is the file.
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_file.pdf|txt|md>")
        sys.exit(1) # Exit with an error code

    # Grab the file path from the terminal command
    file_path = sys.argv[1]

    print(f"[1/4] Loading document: {file_path}")
    # Call our loader function
    raw_text = load_document(file_path)
    print(f"      Loaded {len(raw_text)} characters.")

    print(f"[2/4] Splitting text (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    # Call our chunker function
    chunks = chunk_text(raw_text, source=file_path)
    print(f"      Generated {len(chunks)} chunks.")

    print(f"[3/4] Initializing model ({EMBEDDING_MODEL_NAME}) and embedding...")
    # Load the embedding model into memory (this downloads it from HuggingFace the very first time it runs)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # Call our embedder function
    embeddings = generate_embeddings(chunks, embed_model)

    print(f"[4/4] Writing to persistent vector store at '{PERSIST_DIR}'...")
    # Call our storage function
    total_docs = store_chunks(chunks, embeddings, PERSIST_DIR)
    print(f"      Done! Total records in '{COLLECTION_NAME}': {total_docs}")

    # Quick sanity check verification
    test_query = "What is the battery life and flight time?"
    verify_retrieval(test_query, PERSIST_DIR, embed_model, n_results=2)

if __name__ == "__main__":
    main()
