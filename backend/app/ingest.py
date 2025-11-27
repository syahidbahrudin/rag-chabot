import os, re, hashlib
from typing import List, Dict, Tuple
from .settings import settings

def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _md_sections(text: str) -> List[Tuple[str, str]]:
    # Very simple section splitter by Markdown headings
    parts = re.split(r"\n(?=#+\s)", text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        lines = p.splitlines()
        title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else "Body"
        out.append((title, p))
    return out or [("Body", text)]

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(tokens): break
        i += chunk_size - overlap
    return chunks

def load_documents(data_dir: str) -> List[Dict]:
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            f"Please create the directory and add .md or .txt files, "
            f"or set the DATA_DIR environment variable to point to your data directory."
        )
    if not os.path.isdir(data_dir):
        raise ValueError(f"Data path is not a directory: {data_dir}")
    
    docs = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith((".md", ".txt")):
            continue
        path = os.path.join(data_dir, fname)
        text = _read_text_file(path)
        for section, body in _md_sections(text):
            docs.append({
                "title": fname,
                "section": section,
                "text": body
            })
    return docs

def doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
