#!/usr/bin/env python3
"""
Verification script for Memory v2.0 graph database.
"""

import sqlite3
import numpy as np
import requests

DB_PATH = "/home/boss/vault/memory_graph.db"
OLLAMA_URL = "http://127.0.0.1:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    checks = {
        "total_nodes": c.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
        "total_edges": c.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        "self_node": c.execute("SELECT 1 FROM nodes WHERE id='self:francesca'").fetchone(),
        "user_node": c.execute("SELECT 1 FROM nodes WHERE id='user:flo'").fetchone(),
        "nodes_with_embeddings": c.execute("SELECT COUNT(*) FROM nodes WHERE embedding IS NOT NULL").fetchone()[0],
        "fts_index": c.execute("SELECT COUNT(*) FROM node_fts").fetchone()[0],
        "core_nodes": c.execute("SELECT COUNT(*) FROM nodes WHERE is_core=1").fetchone()[0],
    }

    for name, val in checks.items():
        status = "OK" if val else "FAIL"
        print(f"[{status}] {name}: {val}")

    # Retrieval test: find nodes similar to "market regime"
    test_nodes = c.execute("SELECT id, label, embedding FROM nodes WHERE embedding IS NOT NULL").fetchall()

    r = requests.post(
        OLLAMA_URL,
        json={"model": EMBED_MODEL, "input": "market regime risk off"},
        timeout=60
    )
    r.raise_for_status()
    q_vec = np.array(r.json()["embeddings"][0], dtype=np.float32)

    sims = []
    for nid, label, emb_blob in test_nodes:
        n_vec = np.frombuffer(emb_blob, dtype=np.float32)
        sim = float(np.dot(q_vec, n_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(n_vec)))
        sims.append((sim, nid, label))

    print("\n[RETRIEVAL] Top 3 for 'market regime':")
    for sim, nid, label in sorted(sims, reverse=True)[:3]:
        print(f"  {sim:.3f} | {nid} | {label}")

    # Extra checks
    print("\n[EXTRA] Node type counts:")
    for row in c.execute("SELECT type, COUNT(*) FROM nodes GROUP BY type"):
        print(f"  {row[0]}: {row[1]}")

    print("\n[EXTRA] Edge predicate counts:")
    for row in c.execute("SELECT predicate, COUNT(*) FROM edges GROUP BY predicate"):
        print(f"  {row[0]}: {row[1]}")

    conn.close()
    print("\n[DONE]")


if __name__ == "__main__":
    main()
