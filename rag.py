"""
Phase 4: Real RAG — retrieve relevant chunks, then ask Claude to answer 
using ONLY those chunks. Includes source citations.
"""
import os
from dotenv import load_dotenv
from anthropic import Anthropic
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()
client = Anthropic()

DB_PATH = "./chroma_db"
COLLECTION_NAME = "docs"
TOP_K = 3
MODEL = "claude-sonnet-4-5-20250929"

# RAG-specific system prompt — this is the magic
SYSTEM_PROMPT = """You are a documentation assistant. Answer questions using ONLY the provided context.

Rules:
1. If the answer is in the context, give a clear, concise answer.
2. If the context does NOT contain the answer, respond exactly: "I don't have information about that in the provided documentation."
3. Do not use outside knowledge. Do not guess. Do not make things up.
4. When you answer, cite which source document the information came from."""

# Load embedding model and connect to vector DB
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to vector DB...")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)
print(f"  {collection.count()} chunks indexed\n")


def retrieve(question: str, k: int = TOP_K):
    """Embed the question, return top-k chunks."""
    q_embedding = embed_model.encode([question]).tolist()[0]
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=k,
    )
    chunks = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "similarity": 1 - distance,
        })
    return chunks


def build_context(chunks):
    """Format chunks into a string for the prompt."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {c['source']}]\n{c['text']}")
    return "\n\n".join(parts)


def answer_question(question: str):
    """The main RAG function: retrieve + generate."""
    # 1. Retrieve relevant chunks
    chunks = retrieve(question)
    
    # 2. Build the context
    context = build_context(chunks)
    
    # 3. Build the user message
    user_message = f"""Context from documentation:

{context}

---

Question: {question}

Answer based ONLY on the context above."""
    
    # 4. Send to Claude
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    
    answer = response.content[0].text
    
    # 5. Return everything
    return {
        "question": question,
        "answer": answer,
        "sources": chunks,
        "tokens": {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        },
    }


# Main loop
print("Documentation RAG Assistant — type 'quit' to exit")
print("=" * 60 + "\n")

while True:
    question = input("Q: ").strip()
    if question.lower() == "quit":
        break
    if not question:
        continue
    
    result = answer_question(question)
    
    print(f"\nA: {result['answer']}\n")
    print(f"--- Sources used ---")
    for i, c in enumerate(result["sources"], 1):
        print(f"  {i}. {c['source']} (similarity: {c['similarity']:.3f})")
    print(f"\n--- Tokens: {result['tokens']['input']} in, {result['tokens']['output']} out ---\n")
    print("=" * 60 + "\n")