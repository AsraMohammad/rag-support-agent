"""
Phase 2: Load all markdown files from docs/ and split them into chunks.
Save the chunks to a JSON file we can inspect.
"""
import os
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Where our documents live
DOCS_DIR = Path("docs")
# Where we'll save the chunked output
OUTPUT_FILE = "chunks.json"

# The splitter — this is the workhorse
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # target ~500 characters per chunk
    chunk_overlap=50,      # 50-character overlap between chunks
    separators=["\n\n", "\n", ". ", " ", ""],  # try to split on these, in order
)

all_chunks = []

# Loop through every .md file in docs/
for md_file in DOCS_DIR.glob("*.md"):
    print(f"Loading {md_file.name}...")
    
    # Read the file
    with open(md_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split into chunks
    chunks = splitter.split_text(text)
    print(f"  Split into {len(chunks)} chunks")
    
    # Save each chunk with metadata
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "source": md_file.name,
            "chunk_index": i,
            "text": chunk,
        })

# Write all chunks to a JSON file we can inspect
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"\nTotal chunks: {len(all_chunks)}")
print(f"Saved to: {OUTPUT_FILE}")
print(f"\nFirst chunk preview:")
print(all_chunks[0]["text"][:200])