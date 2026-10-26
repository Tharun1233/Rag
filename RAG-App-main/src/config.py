import os
from dotenv import load_dotenv

load_dotenv()

# Project root (the folder containing src/), so relative paths resolve the
# same no matter which directory the app is launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_project_path(path: str) -> str:
    """Resolves a possibly-relative path against the project root."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


class Config:
    # LLM Settings
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

    # Qdrant Settings
    QDRANT_STORAGE_PATH = _resolve_project_path(
        os.getenv("QDRANT_STORAGE_PATH", "qdrant_storage")
    )
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "pdf_documents")
    
    # Embeddings Settings
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    
    # Text Chunking Settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    @classmethod
    def validate(cls):
        if not cls.OPENROUTER_API_KEY or cls.OPENROUTER_API_KEY == "your_api_key_here":
            raise ValueError("OPENROUTER_API_KEY is not set or is still the default value.")
