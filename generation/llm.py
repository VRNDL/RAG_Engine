from openai import OpenAI

#---Generation---

def generate_answer(query: str, context: str, client: OpenAI, LLM_MODEL: str) -> str:
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
