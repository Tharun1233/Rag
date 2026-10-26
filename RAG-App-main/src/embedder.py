from sentence_transformers import SentenceTransformer
from typing import List
from src.config import Config

class Embedder:
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL_NAME):
        """Initializes the local sentence-transformers model."""
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Generates an embedding for a single query."""
        embedding = self.model.encode([query], show_progress_bar=False)[0]
        return embedding.tolist()
