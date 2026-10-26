"""Gradio web UI for the RAG PDF Assistant (Hugging Face Spaces entry point).

This wrapper only calls the existing, verified pipeline code
(src.pipeline / src.embedder / src.vector_store / src.llm). It changes no
indexing or retrieval logic:

- Upload PDFs -> they are copied into data/ -> RAGPipeline.index_all_pdfs()
- Ask a question -> retrieval runs fully WITHOUT any API key and the
  retrieved chunks (with document + page citations) are always shown.
- The answer/generation step needs the visitor's OWN API key for the
  provider they pick (OpenRouter or Google Gemini), typed into the UI.
  No key is stored, shipped, or read from the server environment for
  visitors.
"""

import os
import shutil

import gradio as gr
import spaces

from src.config import PROJECT_ROOT, Config
from src.pipeline import RAGPipeline

_pipeline = None


def get_pipeline() -> RAGPipeline:
    """One shared pipeline (embedder + local Qdrant) per process."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def _collection_count(pipeline: RAGPipeline) -> int:
    try:
        return pipeline.vector_store.client.count(
            collection_name=Config.QDRANT_COLLECTION_NAME
        ).count
    except Exception:
        return 0


@spaces.GPU(duration=60)
def index_uploaded(files) -> str:
    """Copy uploaded PDFs into data/ and (re)index everything there."""
    pipeline = get_pipeline()
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    copied = []
    for f in files or []:
        path = f if isinstance(f, str) else getattr(f, "name", None)
        if path and path.lower().endswith(".pdf"):
            dest = os.path.join(data_dir, os.path.basename(path))
            if os.path.abspath(path) != os.path.abspath(dest):
                shutil.copyfile(path, dest)
            copied.append(os.path.basename(path))

    pipeline.index_all_pdfs()
    total = _collection_count(pipeline)
    if copied:
        return (
            f"Indexed {', '.join(copied)}. "
            f"{total} chunk(s) now searchable across all PDFs in data/."
        )
    return f"Re-indexed the PDFs already in data/. {total} chunk(s) searchable."


@spaces.GPU(duration=60)
def index_bundled() -> str:
    """Index the PDFs bundled with the repo (data/sample.pdf) on first use."""
    pipeline = get_pipeline()
    if _collection_count(pipeline) == 0:
        pipeline.index_all_pdfs()
    return f"Ready — {_collection_count(pipeline)} chunk(s) indexed from data/."


def _format_chunks(results) -> str:
    if not results:
        return "_No relevant chunks were retrieved._"
    out = []
    for i, doc in enumerate(results, 1):
        meta = doc.get("metadata", {})
        out.append(
            f"**Chunk {i}** — {meta.get('document_name', 'unknown')}, "
            f"page {meta.get('page_number', '?')} "
            f"(score {doc.get('score', 0):.3f})\n\n> {doc.get('text', '')}"
        )
    return "\n\n".join(out)


@spaces.GPU(duration=60)
def ask(question: str, provider: str, api_key: str):
    """Retrieve always; generate only when the visitor supplied a key."""
    question = (question or "").strip()
    if not question:
        return "Type a question first.", ""
    pipeline = get_pipeline()
    if _collection_count(pipeline) == 0:
        pipeline.index_all_pdfs()

    query_emb = pipeline.embedder.embed_query(question)
    results = pipeline.vector_store.search(query_emb, limit=3)
    chunks_md = _format_chunks(results)

    if not results:
        return "No relevant context found in the indexed documents.", chunks_md

    if not (api_key or "").strip():
        return (
            "Retrieval worked — the most relevant chunks are shown below. "
            "To generate an answer from them, pick a provider and add your "
            "own API key above (OpenRouter: openrouter.ai/keys; Google "
            "Gemini: aistudio.google.com/apikey). No key is needed for "
            "retrieval.",
            chunks_md,
        )

    # Use the visitor's own key, in this process only. Nothing is persisted.
    from src.llm import LLMClient

    key = api_key.strip()
    try:
        if provider == "Google Gemini":
            client = LLMClient(
                api_key=key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                model="gemini-2.5-flash",
            )
        else:
            os.environ["OPENROUTER_API_KEY"] = key
            Config.OPENROUTER_API_KEY = key
            client = LLMClient()
        answer = client.generate_answer(question, results)
    except Exception as e:
        answer = (
            f"The answer step failed ({type(e).__name__}: {e}). "
            "Check the API key and try again — retrieval results are below."
        )
    return answer, chunks_md


with gr.Blocks(title="RAG PDF Assistant") as demo:
    gr.Markdown(
        "# RAG PDF Assistant\n"
        "Ask questions over PDF documents and get answers grounded in the "
        "retrieved text, with page-accurate citations. Documents are parsed "
        "and chunked per page, embedded locally (BAAI/bge-small-en-v1.5), "
        "and stored in a local Qdrant vector store."
    )
    status = gr.Markdown()
    demo.load(index_bundled, outputs=status)

    with gr.Row():
        uploads = gr.File(
            label="Upload PDFs",
            file_count="multiple",
            file_types=[".pdf"],
            type="filepath",
        )
    index_btn = gr.Button("Index uploaded PDFs")
    index_status = gr.Markdown()
    index_btn.click(index_uploaded, inputs=uploads, outputs=index_status)

    provider_radio = gr.Radio(
        choices=["OpenRouter", "Google Gemini"],
        value="OpenRouter",
        label="Answer provider (only needed for the answer step)",
    )
    api_key_box = gr.Textbox(
        label="Your API key for the chosen provider (never stored)",
        type="password",
        placeholder="OpenRouter: sk-or-... · Gemini: AIza... — used only for your question",
    )
    question_box = gr.Textbox(
        label="Question",
        placeholder="What is the RAG system?",
    )
    ask_btn = gr.Button("Ask", variant="primary")
    answer_box = gr.Markdown(label="Answer")
    chunks_box = gr.Markdown(label="Retrieved chunks")
    ask_btn.click(ask, inputs=[question_box, provider_radio, api_key_box], outputs=[answer_box, chunks_box])

if __name__ == "__main__":
    demo.launch()
