import os
from typing import List
from src.config import PROJECT_ROOT
from src.pdf_parser import DocumentProcessor
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.llm import LLMClient

class RAGPipeline:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        # Initialize LLM lazily to avoid API key errors during ingestion if not needed
        self.llm = None 

    def index_all_pdfs(self):
        """Scans data/, parses, embeds, and indexes new documents."""
        # Resolve against the project root, not the caller's cwd, so the
        # app finds its documents no matter where it is launched from.
        data_dir = os.path.join(PROJECT_ROOT, "data")
        if not os.path.exists(data_dir):
            print("Data directory not found.")
            return

        all_chunks = []
        failed_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}...")
                    # Isolate per-file failures: one corrupt or unreadable
                    # PDF must not abort indexing of every other document.
                    try:
                        chunks = self.doc_processor.process_pdf(file_path)
                    except Exception as e:
                        failed_files.append(file_path)
                        print(f"Warning: skipping {file_path} ({type(e).__name__}: {e})")
                        continue
                    all_chunks.extend(chunks)
        if failed_files:
            print(f"Skipped {len(failed_files)} unreadable file(s): {', '.join(failed_files)}")
        
        if not all_chunks:
            print("No text could be extracted from the documents.")
            # Drop stale vectors so removed PDFs stop being searchable.
            self.vector_store.clear_collection()
            return

        print(f"Extracted {len(all_chunks)} total chunks. Generating embeddings...")
        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = self.embedder.embed_texts(texts)

        print("Storing in vector database...")
        # Rebuild the collection from scratch on every indexing run so
        # re-running the app never duplicates chunks (idempotent indexing).
        self.vector_store.reset_collection(len(embeddings[0]))
        self.vector_store.index_chunks(all_chunks, embeddings)
        print("Ingestion complete!")

    def query(self, question: str) -> str:
        """Retrieves context for a question and generates an answer."""
        if not self.llm:
            self.llm = LLMClient()
            
        print(f"Generating embedding for query...")
        query_emb = self.embedder.embed_query(question)
        
        print(f"Retrieving relevant context...")
        results = self.vector_store.search(query_emb, limit=3) # Top 3 chunks
        
        if not results:
            return "No relevant context found in the database."
            
        print(f"Context found. Generating answer using LLM...")
        answer = self.llm.generate_answer(question, results)
        
        return answer
