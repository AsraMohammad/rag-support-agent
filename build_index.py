"""
Phase 3a: Build the vector index.
Read chunks.json, embed each chunk, store in ChromaDB.
Run this ONCE after chunking. Re-run if you add more docs.
"""
"""
Phase 3a: Build the vector index from chunks.json.
"""
import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent.resolve()
CHUNKS_FILE = HERE / "chunks.json"
DB_PATH = str(HERE / "chroma_db")
COLLECTION_NAME = "docs"


def build_index():
    """Read chunks.json, embed every chunk, store in ChromaDB."""
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
    
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Embedding chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    
    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    
    collection = client.create_collection(name=COLLECTION_NAME)
    
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {"source": c["source"], "chunk_index": c["chunk_index"]}
            for c in chunks
        ],
    )
    print(f"Stored {len(chunks)} chunks in {DB_PATH}")
    return len(chunks)


if __name__ == "__main__":
    build_index()