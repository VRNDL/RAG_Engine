import heapq
from weaviate_client import get_weaviate_client
from sentence_transformers import CrossEncoder

def retrieve_context(query: str, embed_model, collection_ignored, reranker_model: CrossEncoder, top_k=20, final_k=5) -> str:
    """
    Executes an Alpha-weighted Hybrid Search against Weaviate (Dense + Sparse), 
    followed by Cross-Encoder reranking using a heap for efficient top-k extraction.
    """
    client = get_weaviate_client()
    weaviate_collection = client.collections.get("RAG_Chunks")
    
    # 1. Embed the standalone query for the dense vector portion
    query_vector = embed_model.encode(query).tolist()
    
    # 2. Execute Hybrid Search
    # alpha=0.5 means 50% vector semantic match, 50% exact keyword match (BM25)
    response = weaviate_collection.query.hybrid(
        query=query,
        vector=query_vector,
        alpha=0.5,
        limit=top_k,
        return_properties=["text", "document_context", "source"]
    )
    
    retrieved_docs = []
    for obj in response.objects:
        props = obj.properties
        # Reconstruct the contextualized chunk for the reranker to evaluate
        doc_text = f"[{props.get('document_context', '')}]\n{props.get('text', '')}"
        retrieved_docs.append(doc_text)
        
    client.close()

    # 3. Cross-Encoder Reranking
    if not retrieved_docs:
        return "No relevant context found in the database."
        
    # Pair the query with each retrieved document for the Cross-Encoder
    pairs = [[query, doc] for doc in retrieved_docs]
    scores = reranker_model.predict(pairs)
    
    # Use a heap to efficiently extract the top final_k elements in O(N log K) time
    scored_docs = zip(scores, retrieved_docs)
    top_scored_docs = heapq.nlargest(final_k, scored_docs, key=lambda x: x[0])
    
    # Extract just the document text from the heap results
    best_docs = [doc for score, doc in top_scored_docs]
    
    return "\n\n---\n\n".join(best_docs)