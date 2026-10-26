import os
from typing import List, Dict, Any
from pypdf import PdfReader
from src.config import Config

class DocumentProcessor:
    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, chunk_overlap: int = Config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Reads a PDF and extracts text with metadata per page."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        doc_name = os.path.basename(file_path)
        reader = PdfReader(file_path)
        
        extracted_data = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                extracted_data.append({
                    "text": text,
                    "metadata": {
                        "document_name": doc_name,
                        "page_number": i + 1
                    }
                })
        
        return self._chunk_data(extracted_data)

    def _chunk_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Splits extracted text into chunks while preserving metadata."""
        chunked_data = []
        
        for item in data:
            text = item["text"]
            metadata = item["metadata"]
            
            # Simple recursive-like character splitting (simplified for word boundaries)
            # Split any single word longer than the chunk size (URLs, hashes,
            # base64 blobs) so no chunk can exceed CHUNK_SIZE.
            words = []
            for raw_word in text.split():
                if len(raw_word) > self.chunk_size:
                    words.extend(
                        raw_word[i:i + self.chunk_size]
                        for i in range(0, len(raw_word), self.chunk_size)
                    )
                else:
                    words.append(raw_word)
            current_chunk = []
            current_length = 0
            
            for word in words:
                word_len = len(word) + 1 # +1 for space
                if current_length + word_len > self.chunk_size and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunked_data.append({
                        "text": chunk_text,
                        "metadata": metadata.copy()
                    })
                    
                    # Keep overlap
                    overlap_words = []
                    overlap_len = 0
                    for w in reversed(current_chunk):
                        if overlap_len + len(w) + 1 > self.chunk_overlap:
                            break
                        overlap_words.insert(0, w)
                        overlap_len += len(w) + 1
                        
                    current_chunk = overlap_words
                    current_length = overlap_len
                
                current_chunk.append(word)
                current_length += word_len
                
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunked_data.append({
                    "text": chunk_text,
                    "metadata": metadata.copy()
                })
                
        return chunked_data
