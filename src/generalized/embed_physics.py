import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def embed_physics():
    print("Loading chunks...")
    with open(os.path.join(BASE_DIR, "data/processed/chunks_physics.json"), "r") as f:
        chunks = json.load(f)
    
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "data/chromadb"))
    collection = client.get_or_create_collection("physics")
    
    print(f"Embedding and storing {len(chunks)} chunks...")
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        embeddings = model.encode(batch).tolist()
        collection.add(
            documents=batch,
            embeddings=embeddings,
            ids=[f"physics_chunk_{j}" for j in range(i, i+len(batch))],
            metadatas=[{"source": "openStax_physics", "chunk_index": j} for j in range(i, i+len(batch))]
        )
        print(f"Stored chunks {i} to {i+len(batch)}")
    
    print("Done.")
    
    # test retrieval
    query = "what is Newton's second law"
    embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=3)
    print("\nTest retrieval:")
    for doc in results["documents"][0]:
        print(doc[:200])
        print("---")

if __name__ == "__main__":
    embed_physics()