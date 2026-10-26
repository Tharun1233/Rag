import sys
import src.config # Ensure env is loaded before anything else
from src.pipeline import RAGPipeline

def main():
    print("====================================")
    print("--- RAG PDF Assistant - Interactive CLI ---")
    print("====================================\n")
    
    pipeline = RAGPipeline()
    
    # Run indexing on startup
    print("1. Indexing all PDFs in data/...")
    pipeline.index_all_pdfs()
    print("------------------------------------\n")
    
    print("Type your question below (or 'exit' to quit):")
    while True:
        try:
            query = input("\n> You: ")
            if query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            if not query.strip():
                continue
                
            print("\nThinking...")
            answer = pipeline.query(query)
            print("\n" + "="*40)
            print(answer)
            print("="*40 + "\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
