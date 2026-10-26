from openai import OpenAI
from src.config import Config
from typing import List, Dict, Any

class LLMClient:
    def __init__(self, api_key=None, base_url=None, model=None):
        # Default path (CLI / OpenRouter) is unchanged: key and model come
        # from Config and Config.validate() guards them. A caller may instead
        # pass an explicit key/base_url/model to use another
        # OpenAI-compatible provider (e.g. Google Gemini) — then Config's
        # OpenRouter validation does not apply.
        if api_key is None:
            Config.validate()
            api_key = Config.OPENROUTER_API_KEY
        self.client = OpenAI(
            base_url=base_url or "https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model or Config.OPENROUTER_MODEL

    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generates an answer based on the query and retrieved context."""
        
        context_str = ""
        for idx, doc in enumerate(context_docs):
            doc_name = doc['metadata'].get('document_name', 'Unknown Document')
            page_num = doc['metadata'].get('page_number', 'Unknown Page')
            text = doc['text']
            context_str += f"\n--- Context {idx + 1} ---\nSource: [Document: {doc_name}, Page: {page_num}]\nText: {text}\n"

        system_prompt = """You are a highly precise and helpful AI assistant. 
Your task is to answer the user's question based strictly on the provided context. 

Follow these rules:
1. Answer the question ONLY using the provided retrieved context snippets. If the answer cannot be deduced from the context, strictly output: 'The requested information is not available in the supplied documents.' Do not extrapolate or fabricate.
2. Require the response format:
Answer:
<Clear, concise answer>

Source:
<document_name>
Page <page_number>

Retrieved Text:
"<Exact snippet from the retrieved text that directly proves the answer>"
"""

        user_prompt = f"""Context:
{context_str}

Question:
{query}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0, # Low temperature for more factual and less creative responses
        )

        content = response.choices[0].message.content
        if not content:
            # Some providers/models return an empty completion (content=None);
            # never leak None to callers, which print it as the literal "None".
            return "The model returned an empty response. Please try again."
        return content
