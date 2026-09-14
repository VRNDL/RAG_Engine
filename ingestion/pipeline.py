from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb


#---Loader---

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

def chunk_text(raw_text: str, source: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    splitter=RecursiveCharacterTextSplitter(  #init
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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

def store_chunks(chunks: List[Dict[str, Any]], embeddings: List[List[float]], persist_dir: str, collection_name: str):
    client=chromadb.PersistentClient(path=persist_dir)  # connect to chromadb 
    collection=client.get_or_create_collection(
        name=collection_name,
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