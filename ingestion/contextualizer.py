from openai import OpenAI
from typing import List, Dict, Any

def contextualize_chunks(chunks: List[Dict[str, Any]], full_document_text:str, client: OpenAI, model_name: str) -> List[Dict[str, Any]]:

    # Anthropic prompt template
    prompt_template="""
    <document>
    {WHOLE_DOCUMENT}
    </document>
    Here is the chunk we want to situate within the whole document
    <chunk>
    {CHUNK_CONTENT}
    </chunk>
    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else
    """ 

    print(f" -> Contextualising {len(chunks)} chunks")

    for idx, chunk in enumerate(chunks):
        # what goes into the prompt
        prompt=prompt_template.format(
            WHOLE_DOCUMENT=full_document_text,
            CHUNK_CONTENT=chunk["text"]
        )

        response=client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user","content":"prompt"}],
            temperature=0.0
        )

        situating_context=response.choices[0].message.content.strip()

        # prepend generated context to existing chunk
        original=chunk["text"]
        chunk["text"]=f"{situating_context}\n\n{original}"


        # add new column to metadata
        chunk["metadata"]["situating_context"]=situating_context

        print(f"    - Chunk {idx+1}/{len(chunks)} contextualized.")
        
    return chunks