import sys
import time
import src.config
from src.pipeline import RAGPipeline

def run_demo():
    print("====================================")
    print("--- Automated Demo Runner ---")
    print("====================================\n")
    
    pipeline = RAGPipeline()
    
    print("Step 1: Indexing all PDFs in data/...")
    pipeline.index_all_pdfs()
    print("Indexing complete.\n")
    
    questions = [
        "What is the RAG system?",
        "What are the benefits of vector embeddings?",
        "How does semantic search work according to the document?",
        "What is the capital of France?" # Out of domain question
    ]
    
    for i, q in enumerate(questions):
        print(f"\n--- Test Query {i+1} ---")
        print(f"> You: {q}")
        print("Thinking...")
        try:
            start = time.time()
            answer = pipeline.query(q)
            duration = time.time() - start
            
            print("\n" + "="*40)
            print(answer)
            print("="*40)
            print(f"(Answered in {duration:.2f}s)\n")
            
        except Exception as e:
            print(f"\nError occurred: {str(e)}\n")
        
        time.sleep(1) # Small pause for readability in demo

if __name__ == "__main__":
    run_demo()
