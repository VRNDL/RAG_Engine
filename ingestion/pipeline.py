from pathlib import Path
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer
from weaviate_client import get_weaviate_client, setup_schema


#---Loader---

def load_document(file_path: str) -> str:
    """Converts PDF into markdown via library"""
    path=Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f" -> Converting {path.name} to Markdown using Docling...")
    converter=DocumentConverter()
    result=converter.convert(file_path)     # actual conversion
    return result.document.export_to_markdown()



#--Chunking--

def chunk_text(raw_markdown: str, source: str) -> List[Dict[str, Any]]:
    """Splits mardown logically based on headers"""
    headers_to_split_on=[     # what to split on
        ("#","Header 1"),
        ("##","Header 2"),
        ("###","Header 3"),
    ]

    # actual splitting
    markdown_splitter=MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)  # rule
    md_splits=markdown_splitter.split_text(raw_markdown)                                   # splitting text on headers as list of documents


    # if a single section is still massive (yk what else is massive? THE LOWWW TA-)
    char_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_splits=char_splitter.split_documents(md_splits)                       # metadata is copied onto the split files as list of documents
    chunks=[]

    for idx, doc in enumerate(final_splits):
        meta=doc.metadata.copy()     # doc.metadata now contains section headers
        meta["source"]=str(source)   # attaching new column called source that has the file path
        meta["chunk_index"]=idx
        chunks.append({                              # putting dicts in a list
            "id" : f"{Path(source).stem}_md_{idx}",     # file name and index as key
            "text" : doc.page_content,
            "metadata" : meta
        })
    return chunks


#--Embedding--

def generate_embeddings(chunks: List[Dict[str, Any]], model: SentenceTransformer):
    texts=[c["text"] for c in chunks]  # extract raw text from the dicts
    # scaled for cosine similarity
    embeddings=model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.tolist()  # numpy to list


#--Storage of chunks--

def store_chunks(chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
    """Pushes chunks and vectors to weaviate for hybrid search"""
    client=get_weaviate_client()
    collection=setup_schema(client)

    with collection.batch.dynamic() as batch:
        for idx, chunk in enumerate(chunks):
            meta=chunk["metadata"]
            context_parts=[]
            for header in ["Header 1", "Header 2", "Header 3"]:
                if header in meta:
                    context_parts.append(f"{header}: {meta[header]}")

            document_context=" | ".join(context_parts) if context_parts else "No headers"

            batch.add_object(
                properties={
                    "text": chunk["text"],
                    "document_context": document_context,
                    "source": meta.get("source","Unknown")
                },
                vector=embeddings[idx]
            )

    print("Successfully built Weaviate Hybrid Indexes!")
    client.close()