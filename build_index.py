"""
Phase 3a: Build the vector index.
Read chunks.json, embed each chunk, store in ChromaDB.
Run this ONCE after chunking. Re-run if you add more docs.
"""
import json
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "chunks.json"
DB_PATH = "./chroma_db"  # ChromaDB will save its files here
COLLECTION_NAME = "docs"

print("Loading chunks...")
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"  Loaded {len(chunks)} chunks")

# Load the embedding model (downloads ~80MB the first time)
print("\nLoading embedding model (first run downloads ~80MB)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("  Model loaded")

# Embed every chunk
print("\nEmbedding chunks...")
texts = [c["text"] for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True).tolist()
print(f"  Generated {len(embeddings)} embeddings")
print(f"  Each embedding has {len(embeddings[0])} dimensions")

# Set up ChromaDB
print("\nSetting up ChromaDB...")
client = chromadb.PersistentClient(path=DB_PATH)

# If the collection exists from a prior run, delete it (so we don't duplicate)
try:
    client.delete_collection(name=COLLECTION_NAME)
    print("  Deleted existing collection")
except Exception:
    pass  # didn't exist yet, that's fine

collection = client.create_collection(name=COLLECTION_NAME)
print(f"  Created collection: {COLLECTION_NAME}")

# Insert chunks into the collection
print("\nStoring in vector DB...")
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=texts,
    embeddings=embeddings,
    metadatas=[
        {"source": c["source"], "chunk_index": c["chunk_index"]}
        for c in chunks
    ],
)
print(f"  Stored {len(chunks)} chunks")

print(f"\nDone. Vector DB saved to: {DB_PATH}")