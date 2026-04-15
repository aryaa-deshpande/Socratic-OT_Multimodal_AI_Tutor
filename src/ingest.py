from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import fitz
import os

def save_chunks(chunks):
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/chunks.json", "w") as f:
        json.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks to data/processed/chunks.json")

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def clean_text(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if len(line) < 80:  
            continue
        if any(char.isdigit() and len(line) < 100 for char in line[:5]):
            continue 
        cleaned.append(line)
    return " ".join(cleaned)

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    return chunks


if __name__ == "__main__":
    text = extract_text("data/raw/Anatomy_and_Physiology_2e.pdf")
    print(f"Extracted {len(text)} characters")
    print(text[:500]) 

    print("Extracting Text")
    text = extract_text("data/raw/Anatomy_and_Physiology_2e.pdf")
    
    print("Cleaning Text")
    cleaned = clean_text(text)
    
    print("Chunking Text")
    chunks = chunk_text(cleaned)
    
    print(f"Total chunks: {len(chunks)}")
    print("\n Sample chunk ")
    print(chunks[10])

    print("Saving chunks")
    save_chunks(chunks)

    print("\n--- Sample chunks ---")
    print("Chunk 100:", chunks[100])
    print("\nChunk 500:", chunks[500])
    print("\nChunk 1000:", chunks[1000])

    



    