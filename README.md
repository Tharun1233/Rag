# RAG PDF Assistant


## 1. Architecture Overview
This application implements a modular Retrieval-Augmented Generation (RAG) pipeline:
**Parser $\rightarrow$ Embedder $\rightarrow$ Qdrant $\rightarrow$ Retrieval $\rightarrow$ OpenRouter LLM**

1. **Parser**: Recursively scans the `data/` folder for PDFs, extracts text per page using `pypdf`, and chunks it (with overlap) while preserving exact `document_name` and `page_number` metadata to ensure citations are 100% accurate.
2. **Embedder**: Generates vector representations of the chunks locally using the `sentence-transformers` library to avoid API costs and latency.
3. **Qdrant**: Stores the embeddings alongside their metadata in a local Qdrant vector database (`qdrant_storage/`), allowing for fast, persistent similarity search without requiring Docker.
4. **Retrieval**: When a query is asked, it is embedded using the same local model, and the top-k most semantically similar chunks are retrieved from Qdrant using Cosine distance.
5. **OpenRouter LLM**: The retrieved context is formatted into a strict system prompt and sent to a free LLM via OpenRouter. The LLM is forced to answer ONLY based on the context, preventing hallucinations and formatting the output cleanly with direct citations.

## 2. Libraries Used
- **`pypdf`**: Robust PDF text extraction.
- **`sentence-transformers`**: Local generation of semantic embeddings.
- **`qdrant-client`**: Vector database client operating in local disk mode.
- **`openai`**: Official OpenAI SDK configured to point to the OpenRouter API for LLM completion.
- **`python-dotenv`**: Environment variable management.

## 3. Embedding & LLM Models Used
- **Embedding Model**: `BAAI/bge-small-en-v1.5` (Local execution, 384 dimensions, extremely fast and accurate).
- **LLM Provider**: `meta-llama/llama-3.3-70b-instruct:free` via OpenRouter (Can be configured in `.env`).

## 4. Assumptions Made
- **Chunking**: Chunks are constructed strictly within page boundaries. A single chunk will never span multiple pages or multiple documents. This guarantees that citations refer to an exact page.
- **File Structure**: PDFs are flatly stored or organized in folders strictly under the `data/` directory.
- **API Availability**: OpenRouter API is accessible and valid.

## 5. Setup & Execution Guide

### Initial Setup
```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API Key
# Copy .env.example to .env and insert your OpenRouter API Key
# OPENROUTER_API_KEY=your_key_here

# 4. Add Documents
# Place your PDFs inside the `data/` folder.
```

### Execution
You can run the application in two ways:

1. **Interactive Chat CLI**
```bash
python main.py
```
*This will index any PDFs and open an interactive prompt for you to query.*

2. **Automated Demo Runner**
```bash
python demo.py
```
*This automatically indexes PDFs and runs 4 pre-configured test queries (including one out-of-domain query to prove hallucination prevention) with clean output tailored for a demo recording.*
