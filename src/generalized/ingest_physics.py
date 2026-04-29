import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest import extract_text, clean_text, chunk_text
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    pdf_path = os.path.join(BASE_DIR, "data/raw/Physics.pdf")
    
    print("Extracting")
    text = extract_text(pdf_path)
    
    print("Cleaning")
    cleaned = clean_text(text)
    
    print("Chunking")
    chunks = chunk_text(cleaned)
    print(f"Total chunks: {len(chunks)}")
    
    output_path = os.path.join(BASE_DIR, "data/processed/chunks_physics.json")
    with open(output_path, "w") as f:
        json.dump(chunks, f)
    print(f"Saved to data/processed/chunks_physics.json")