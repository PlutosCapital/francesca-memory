#!/usr/bin/env python3
"""
Icarus Lightweight Indexer — Ollama nomic-embed-text + numpy + sklearn
No FAISS. No sentence-transformers. Free local embeddings.
"""

import os
import sys
import json
import glob
import pickle
import requests
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL = "nomic-embed-text"

VAULT = Path("/home/boss/vault")
INDEX_DIR = VAULT / "icarus" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = INDEX_DIR / "icarus_vectors.pkl"
META_FILE = INDEX_DIR / "icarus_meta.json"

SOURCES = [
    VAULT / "L1-working" / "*.md",
    VAULT / "L2-episodic" / "daily" / "*.md",
    VAULT / "L2-episodic" / "scribe" / "*.md",
    VAULT / "L3-semantic" / "identity" / "*.md",
    VAULT / "L3-semantic" / "knowledge" / "*.json",
]

def embed(text):
    r = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBED_MODEL, "input": text[:512]},
        timeout=60,
    )
    r.raise_for_status()
    return np.array(r.json()["embeddings"][0], dtype=np.float32)

def build_index():
    print("[INDEX] Scanning vault...")
    docs = []
    for pattern in SOURCES:
        for path in glob.glob(str(pattern)):
            p = Path(path)
            content = p.read_text()
            body = content.split("---")[-1].strip() if "---" in content else content.strip()
            docs.append({"path": str(p.relative_to(VAULT)), "body": body[:1000], "mtime": p.stat().st_mtime})
    
    print(f"[INDEX] {len(docs)} documents")
    if not docs:
        print("[WARN] No docs"); return
    
    vectors, meta = [], []
    for i, doc in enumerate(docs):
        try:
            vec = embed(doc["body"])
            vectors.append(vec)
            meta.append(doc)
            print(f"  [{i+1}/{len(docs)}] {doc['path']}")
        except Exception as e:
            print(f"  [FAIL] {doc['path']}: {e}")
    
    if not vectors:
        print("[ERROR] No embeddings"); return
    
    matrix = np.vstack(vectors)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(matrix, f)
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INDEX] Saved: {matrix.shape[0]} vectors, {matrix.shape[1]} dims")

def search(query, top_k=5):
    if not INDEX_FILE.exists():
        print("[ERROR] Run --build first"); return []
    with open(INDEX_FILE, "rb") as f:
        matrix = pickle.load(f)
    with open(META_FILE, "r") as f:
        meta = json.load(f)
    q_vec = embed(query).reshape(1, -1)
    sims = cosine_similarity(q_vec, matrix)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [{"path": meta[i]["path"], "score": float(sims[i]), "snippet": meta[i]["body"][:200]} for i in top_idx]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        build_index()
    elif len(sys.argv) > 1 and sys.argv[1] == "--search":
        query = sys.argv[2] if len(sys.argv) > 2 else "market regime"
        for r in search(query):
            print(f"\n[{r['score']:.3f}] {r['path']}\n  {r['snippet'][:150]}...")
    else:
        print("Usage: python3 icarus_indexer.py --build | --search 'query'")
