from openai import OpenAI
from typing import List, Dict

def formulate_query(current_query: str, chat_history: List[Dict[str, str]], client: OpenAI, model_name: str) -> str:
    if not chat_history:
        return current_query

    # format last 6 messages for the LLM
    history_text="\n".join([f"{msg['role'].capitalize()}: {msg,['content']}" for msg in chat_history])

    prompt=f"""Given the following conversation history, rewrite the user's latest query to be completely self-contained and standalone.
                If it is already standalone, return the original query.
                Do NOT answer the question. ONLY return the rewritten query.

                History:
                {history_text}

                Latest Query: {current_query}"""

    response=client.chat.completions.create(
        model=model_name,
        messages=[{"role":"user", "content":prompt}],
        temperature=0.0
    )

    return response.choices[0].message.content.strip()