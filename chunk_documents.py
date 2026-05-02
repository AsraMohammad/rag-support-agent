"""
Phase 2: Load all markdown files from docs/ and split them into chunks.
Save the chunks to a JSON file we can inspect.
"""
"""
Phase 2: Load all markdown files from docs/ and split them into chunks.
"""
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use absolute path resolution so it works regardless of cwd
HERE = Path(__file__).parent.resolve()
DOCS_DIR = HERE / "docs"
OUTPUT_FILE = HERE / "chunks.json"


def chunk_all_documents():
    """Read every .md file in docs/, split into chunks, save to chunks.json."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    all_chunks = []
    md_files = list(DOCS_DIR.glob("*.md"))
    
    if not md_files:
        raise RuntimeError(f"No .md files found in {DOCS_DIR}")
    
    for md_file in md_files:
        print(f"Loading {md_file.name}...")
        with open(md_file, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = splitter.split_text(text)
        print(f"  Split into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": md_file.name,
                "chunk_index": i,
                "text": chunk,
            })
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"Total chunks: {len(all_chunks)}")
    return len(all_chunks)


if __name__ == "__main__":
    n = chunk_all_documents()
    print(f"Saved to: {OUTPUT_FILE}")