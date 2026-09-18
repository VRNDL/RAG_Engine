from openai import OpenAI
from typing import List, Dict

#---Generation---

def generate_answer(query: str, context: str, chat_history: List[Dict[str, str]], client: OpenAI, LLM_MODEL: str) -> str:
    # System prompt: tells LLM how to behave
    system_prompt = """
    You are an expert technical assistant.
    Answer the user's question using ONLY the provided context below.
    If the context does not contain the answer, say "I cannot answer this based on the provided documents."
    Do not use outside knowledge. Be concise and direct.
    
    CONTEXT:
    {context}
    """

    # messages extracted so that we can sandwich it between system prompt and new formatted prompt

    messages=[{"role":"system", "content":system_prompt}]

    for msg in chat_history:
        messages.append(msg)

    # format the prompt putting retireved chunks in {context}
    final_prompt=f"Context:\n{context}\n\nQuestion: {query}"
    messages.append({"role":"user", "content": final_prompt})

    # call LLM
    response=client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.0    # creativity level
    )


    # return AI generated text
    return response.choices[0].message.content
