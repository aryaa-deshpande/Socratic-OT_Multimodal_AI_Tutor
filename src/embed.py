import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_chunks():
    path = os.path.join(BASE_DIR, "data/processed/chunks.json")
    with open(path, "r") as f:
        return json.load(f)

def embed_and_store(chunks):
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Setting up ChromaDB")
    client = chromadb.PersistentClient(
        path=os.path.join(BASE_DIR, "data/chromadb")
    )
    collection = client.get_or_create_collection("anatomy")
    
    print(f"Embedding and storing {len(chunks)} chunks...")
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        embeddings = model.encode(batch).tolist()
        collection.add(
            documents=batch,
            embeddings=embeddings,
            ids=[f"chunk_{j}" for j in range(i, i+len(batch))],
            metadatas=[{"source": "openStax_anatomy", "chunk_index": j} for j in range(i, i+len(batch))]
        )
        print(f"Stored chunks {i} to {i+len(batch)}")
    
    print("Done.")
    return collection

def test_retrieval(collection):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query = "what spinal levels form the brachial plexus"
    embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=3)
    print("\n Test retrieval:")
    for doc in results["documents"][0]:
        print(doc[:300])
        print("---")

if __name__ == "__main__":
    chunks = load_chunks()
    collection = embed_and_store(chunks)
    test_retrieval(collection)

    