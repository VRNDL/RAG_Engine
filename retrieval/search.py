from sentence_transformers import SentenceTransformer, CrossEncoder
import heapq

def retrieve_context(query: str, model: SentenceTransformer, collection, reranker: CrossEncoder, num_results: int=15, top: int=3)-> str:
    """Embed user's query and find chunk most relevant to it"""
    query_embedding=model.encode([query], normalize_embeddings=True).tolist()   # turn query into vector

    #search the db
    results=collection.query(
        query_embeddings=query_embedding,
        n_results=num_results
    )

    # extract text chunks (since we are passing only 1 query, we will get the 3 nearest chunks for that query at index 0)
    retrieved_chunks=results["documents"][0]

    # model needs pair [query, chunk_i]
    pairs=[[query, chunk] for chunk in retrieved_chunks]

    # generate relevance score for each pair
    score=reranker.predict(pairs)

    #sort them via heap
    top_scored_pairs=heapq.nlargest(top, zip(score, retrieved_chunks), key=lambda x:x[0])  

    # join the chunks separated by ---
    context_string="\n\n---\n\n".join([chunk for _, chunk in top_scored_pairs])
    return context_string