"""
Phase 3b: Search the vector DB.
Take a question, find the most relevant chunks.
"""
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "./chroma_db"
COLLECTION_NAME = "docs"
TOP_K = 3  # how many chunks to return

# Load the same embedding model used to build the index
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to the existing ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
print(f"Connected to collection: {COLLECTION_NAME}")
print(f"  Total chunks indexed: {collection.count()}\n")

while True:
    question = input("Ask a question (or 'quit'): ").strip()
    if question.lower() == "quit":
        break
    if not question:
        continue
    
    # Embed the question
    question_embedding = model.encode([question]).tolist()[0]
    
    # Search for the top K most similar chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=TOP_K,
    )
    
    # Display results
    print(f"\nTop {TOP_K} results:\n")
    for i, (doc, meta, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        # ChromaDB returns distance (lower = more similar). Convert to similarity score.
        similarity = 1 - distance
        print(f"--- Result {i+1} (similarity: {similarity:.3f}, source: {meta['source']}) ---")
        print(doc[:300])  # first 300 chars
        print()
    print("=" * 60 + "\n")