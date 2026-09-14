import os
import sys

from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
from dotenv import load_dotenv


PERSIST_DIR="./chroma_db"
COLLECTION_NAME="rag_engine_baseline"
EMBEDDING_MODEL_NAME="BAAI/bge-small-en-v1.5"
LLM_MODEL="openai/gpt-oss-120b"
NUM_RESULTS=3      # how many chunks to feed to the LLM

load_dotenv()   # load variables from .env file

#--Retrieval--

def retrieve_context(query: str, model: SentenceTransformer, collection) -> str:
    """Embed user's query and find chunk most relevant to it"""
    query_embedding=model.encode([query], normalize_embeddings=True).tolist()   # turn query into vector

    #search the db
    results=collection.query(
        query_embeddings=query_embedding,
        n_results=NUM_RESULTS
    )

    # extract text chunks (since we are passing only 1 query, we will get the 3 nearest chunks for that query at index 0)
    retrieved_chunks=results["documents"][0]

    # join the chunks separated by ---
    context_string="\n\n---\n\n".join(retrieved_chunks)
    return context_string


#---Generation---

def generate_answer(query: str, context: str, client: OpenAI) -> str:
    # System prompt: tells LLM how to behave
    system_prompt = """
    You are an expert technical assistant.
    Answer the user's question using ONLY the provided context below.
    If the context does not contain the answer, say "I cannot answer this based on the provided documents."
    Do not use outside knowledge. Be concise and direct.
    
    CONTEXT:
    {context}
    """

    # format the prompt putting retireved chunks in {context}
    formatted_system_prompt=system_prompt.format(context=context)

    # call LLM
    response=client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": formatted_system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.0    # creativity level
    )


    # return AI generated text
    return response.choices[0].message.content


# --- Execution Orchestration ---
def main():
    # Make sure the user actually typed a question in the terminal
    if len(sys.argv) < 2:
        print('Usage: python query.py "Your question here"')
        sys.exit(1)
        
    user_query = sys.argv[1]
    
    print("[1/4] Loading database and models...")
    # Connect to the Chroma database we built in Step 1
    db_client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = db_client.get_collection(name=COLLECTION_NAME)
    
    # Load the embedding model
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Initialize the OpenAI client pointing to Groq's API endpoint
    llm_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    
    print("[2/4] Searching database for relevant context...")
    context = retrieve_context(user_query, embed_model, collection)
    
    print("[3/4] Generating answer...")
    answer = generate_answer(user_query, context, llm_client)
    
    print("\n==========================================")
    print(f"QUESTION: {user_query}")
    print("==========================================\n")
    print(answer)
    print("\n==========================================")

if __name__ == "__main__":
    main()