from sentence_transformers import SentenceTransformer

def retrieve_context(query: str, model: SentenceTransformer, collection, num_results: int=3)-> str:
    """Embed user's query and find chunk most relevant to it"""
    query_embedding=model.encode([query], normalize_embeddings=True).tolist()   # turn query into vector

    #search the db
    results=collection.query(
        query_embeddings=query_embedding,
        n_results=num_results
    )

    # extract text chunks (since we are passing only 1 query, we will get the 3 nearest chunks for that query at index 0)
    retrieved_chunks=results["documents"][0]

    # join the chunks separated by ---
    context_string="\n\n---\n\n".join(retrieved_chunks)
    return context_string