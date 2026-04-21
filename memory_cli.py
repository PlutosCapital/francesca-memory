#!/usr/bin/env python3
"""
Memory CLI — Admin tool for the v2.0 graph memory system.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import requests

DB_PATH = Path("/home/boss/vault/memory_graph.db")
EMBED_URL = "http://127.0.0.1:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


def status():
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    checks = {
        "nodes": c.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
        "edges": c.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        "events_total": c.execute("SELECT COUNT(*) FROM events").fetchone()[0],
        "events_unprocessed": c.execute("SELECT COUNT(*) FROM events WHERE processed=0").fetchone()[0],
        "working_memory": c.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0],
        "core_nodes": c.execute("SELECT COUNT(*) FROM nodes WHERE is_core=1").fetchone()[0],
        "fts_index": c.execute("SELECT COUNT(*) FROM node_fts").fetchone()[0],
    }
    conn.close()
    print("=== Memory Graph Status ===")
    for k, v in checks.items():
        print(f"  {k:20s}: {v}")


def query(text: str, top_k: int = 5):
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    r = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "input": text[:512]},
        timeout=60,
    )
    r.raise_for_status()
    q_vec = np.array(r.json()["embeddings"][0], dtype=np.float32)

    rows = c.execute("SELECT id, type, label, content, embedding FROM nodes WHERE embedding IS NOT NULL").fetchall()
    sims = []
    for nid, ntype, label, content, emb_blob in rows:
        n_vec = np.frombuffer(emb_blob, dtype=np.float32)
        sim = float(np.dot(q_vec, n_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(n_vec)))
        sims.append((sim, nid, ntype, label, content[:120]))

    print(f"=== Query: '{text}' ===")
    for sim, nid, ntype, label, content in sorted(sims, reverse=True)[:top_k]:
        print(f"\n  {sim:.3f} | {nid} ({ntype})")
        print(f"    {label}")
        print(f"    {content}...")

    conn.close()


def inspect(node_id: str):
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT * FROM nodes WHERE id=?", (node_id,))
    row = c.fetchone()
    if not row:
        print(f"Node {node_id} not found.")
        conn.close()
        return

    cols = [d[0] for d in c.description]
    print(f"=== Node: {node_id} ===")
    for col, val in zip(cols, row):
        if col == "embedding":
            val = f"<{len(val) if val else 0} bytes>"
        print(f"  {col:20s}: {val}")

    print("\n  -- Outgoing edges --")
    for e in c.execute("SELECT predicate, target_id, weight, confidence FROM edges WHERE source_id=?", (node_id,)):
        print(f"    --[{e[0]}]--> {e[1]} (w={e[2]:.2f}, c={e[3]:.2f})")

    print("\n  -- Incoming edges --")
    for e in c.execute("SELECT predicate, source_id, weight, confidence FROM edges WHERE target_id=?", (node_id,)):
        print(f"    {e[1]} --[{e[0]}]--> (w={e[2]:.2f}, c={e[3]:.2f})")

    conn.close()


def inject(event_type: str, payload: str):
    import uuid as uuid_mod
    from datetime import datetime, timezone
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    eid = f"manual:{uuid_mod.uuid4().hex[:16]}"
    now = datetime.now(timezone.utc).isoformat()
    c.execute("""
        INSERT INTO events (id, timestamp, event_type, payload, processed)
        VALUES (?, ?, ?, ?, 0)
    """, (eid, now, event_type, payload))
    conn.commit()
    conn.close()
    print(f"Injected event {eid}")


def working():
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    rows = c.execute("""
        SELECT wm.node_id, n.type, n.label, n.activation, wm.access_count, wm.last_accessed
        FROM working_memory wm
        JOIN nodes n ON wm.node_id = n.id
        ORDER BY wm.last_accessed DESC
    """).fetchall()
    print(f"=== Working Memory ({len(rows)} nodes) ===")
    for nid, ntype, label, activation, access_count, last_accessed in rows:
        print(f"  {activation:.2f} | {access_count:3d} | {nid:40s} | {label}")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Memory CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("status", help="Graph statistics")
    sub.add_parser("working", help="Working memory contents")

    p_query = sub.add_parser("query", help="Semantic query")
    p_query.add_argument("text", help="Query text")
    p_query.add_argument("--top-k", type=int, default=5)

    p_inspect = sub.add_parser("inspect", help="Inspect a node")
    p_inspect.add_argument("node_id", help="Node ID")

    p_inject = sub.add_parser("inject", help="Inject a manual event")
    p_inject.add_argument("--type", default="user_message", choices=["file_write","api_error","user_message","agent_output","tool_call","system_change","decision"])
    p_inject.add_argument("payload", help="Event payload (text or JSON)")

    args = parser.parse_args()
    if args.cmd == "status":
        status()
    elif args.cmd == "working":
        working()
    elif args.cmd == "query":
        query(args.text, args.top_k)
    elif args.cmd == "inspect":
        inspect(args.node_id)
    elif args.cmd == "inject":
        inject(args.type, args.payload)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
