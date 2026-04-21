#!/usr/bin/env python3
"""
Memory Hippocampus — Phase 1 ingestion engine.
Polls unprocessed events, scores salience (Ollama), extracts entities,
ingests into graph with semantic dedup, updates working memory.
"""

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

DB_PATH = Path("/home/boss/vault/memory_graph.db")
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "francesca-local"
EMBED_URL = "http://127.0.0.1:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"

SALIENCE_THRESHOLD = 0.6
SIMILARITY_MERGE = 0.92
SIMILARITY_LINK = 0.75
MAX_EVENTS_PER_RUN = 10
MAX_PAYLOAD_LEN = 1500


def ollama_generate(prompt: str, system: str = None, timeout: int = 60) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "5m",
    }
    if system:
        payload["system"] = system
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"]


def embed_text(text: str) -> bytes:
    payload = text[:512] if text else ""
    r = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "input": payload},
        timeout=60,
    )
    r.raise_for_status()
    vec = np.array(r.json()["embeddings"][0], dtype=np.float32)
    return vec.tobytes()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def parse_json_from_text(text: str) -> dict:
    """Extract JSON from potentially markdown-wrapped text."""
    text = text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON object
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def score_salience(payload_text: str) -> dict:
    """Returns {surprise, cost, identity, max_score}."""
    prompt = f"""You are a salience scorer for an AI agent's memory system.
Score this event on three dimensions (0.0 to 1.0):
- surprise: how unexpected is this? (0=routine, 1=shocking)
- cost: what is at stake? (0=nothing, 1=critical failure or expensive mistake)
- identity: does this change what the agent knows about itself or its user? (0=irrelevant, 1=core identity change)

Output ONLY a JSON object: {{"surprise": float, "cost": float, "identity": float}}

Event:
---
{payload_text[:MAX_PAYLOAD_LEN]}
---
"""
    try:
        raw = ollama_generate(prompt, system="You output only valid JSON. No extra text.")
        parsed = parse_json_from_text(raw)
        s = float(parsed.get("surprise", 0))
        co = float(parsed.get("cost", 0))
        i = float(parsed.get("identity", 0))
        return {"surprise": s, "cost": co, "identity": i, "max_score": max(s, co, i)}
    except Exception as e:
        print(f"[WARN] Salience scoring failed: {e}")
        return {"surprise": 0, "cost": 0, "identity": 0, "max_score": 0}


def extract_entities(payload_text: str) -> dict:
    """Returns {entities: [{name, type}], relations: [{subject, predicate, object}]}."""
    prompt = f"""Extract entities and relations from this event.
Entities: list of {{"name": str, "type": str}} where type is one of [file, api, person, concept, decision, error, behavior]
Relations: list of {{"subject": str, "predicate": str, "object": str}} where predicate is one of [caused_by, part_of, uses, depends_on, predicts, authored]

Output ONLY JSON: {{"entities": [...], "relations": [...]}}

Event:
---
{payload_text[:MAX_PAYLOAD_LEN]}
---
"""
    try:
        raw = ollama_generate(prompt, system="You output only valid JSON. No extra text.", timeout=90)
        return parse_json_from_text(raw)
    except Exception as e:
        print(f"[WARN] Entity extraction failed: {e}")
        return {"entities": [], "relations": []}


def get_all_embeddings(conn: sqlite3.Connection) -> list:
    c = conn.cursor()
    c.execute("SELECT id, embedding FROM nodes WHERE embedding IS NOT NULL")
    rows = c.fetchall()
    result = []
    for nid, emb_blob in rows:
        vec = np.frombuffer(emb_blob, dtype=np.float32)
        result.append((nid, vec))
    return result


def semantic_dedup(conn: sqlite3.Connection, new_emb: bytes, existing: list) -> tuple:
    """Returns (action, node_id_or_none). action = 'merge', 'link', or 'new'."""
    new_vec = np.frombuffer(new_emb, dtype=np.float32)
    best_sim = -1
    best_id = None
    for nid, vec in existing:
        sim = cosine_sim(new_vec, vec)
        if sim > best_sim:
            best_sim = sim
            best_id = nid

    if best_sim >= SIMILARITY_MERGE:
        return ("merge", best_id, best_sim)
    elif best_sim >= SIMILARITY_LINK:
        return ("link", best_id, best_sim)
    return ("new", None, 0.0)


def node_type_from_entity(etype: str) -> str:
    mapping = {
        "file": "file",
        "api": "api",
        "person": "user",
        "concept": "concept",
        "decision": "decision",
        "error": "error",
        "behavior": "behavior",
    }
    return mapping.get(etype, "concept")


def upsert_node(conn: sqlite3.Connection, node_id: str, node_type: str, label: str,
                content: str, activation: float, source: str, is_core: bool = False):
    c = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    tc = len(content.split())
    c.execute("""
        INSERT OR REPLACE INTO nodes
        (id, type, label, content, activation, created_at, updated_at, source, is_core, token_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (node_id, node_type, label, content, activation, now, now, source, int(is_core), tc))
    conn.commit()


def update_node_embedding(conn: sqlite3.Connection, node_id: str, emb: bytes):
    c = conn.cursor()
    c.execute("UPDATE nodes SET embedding=? WHERE id=?", (emb, node_id))
    conn.commit()
    # Sync FTS
    c.execute("SELECT content FROM nodes WHERE id=?", (node_id,))
    row = c.fetchone()
    if row:
        c.execute("INSERT OR REPLACE INTO node_fts (rowid, content) VALUES ((SELECT rowid FROM nodes WHERE id=?), ?)", (node_id, row[0]))
    conn.commit()


def insert_edge(conn: sqlite3.Connection, edge_id: str, source_id: str, predicate: str,
                target_id: str, weight: float = 1.0, confidence: float = 0.75):
    c = conn.cursor()
    c.execute("SELECT 1 FROM edges WHERE id=?", (edge_id,))
    if c.fetchone():
        return False
    now = datetime.now(timezone.utc).isoformat()
    c.execute("""
        INSERT INTO edges (id, source_id, target_id, predicate, weight, confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (edge_id, source_id, target_id, predicate, weight, confidence, now))
    conn.commit()
    return True


def update_working_memory(conn: sqlite3.Connection, node_id: str):
    c = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    c.execute("""
        INSERT INTO working_memory (node_id, inserted_at, access_count, last_accessed)
        VALUES (?, ?, 1, ?)
        ON CONFLICT(node_id) DO UPDATE SET
            access_count = access_count + 1,
            last_accessed = excluded.last_accessed
    """, (node_id, now, now))
    conn.commit()


def ingest_event(conn: sqlite3.Connection, event: tuple, existing_embeddings: list):
    """Process a single event. event = (id, event_type, payload)."""
    eid, etype, payload = event
    print(f"\n[HIPPO] Processing event {eid[:8]}... ({etype})")

    # Parse payload
    try:
        pdata = json.loads(payload)
        text = pdata.get("snippet", payload)
        path = pdata.get("path", "")
    except json.JSONDecodeError:
        text = payload
        path = ""

    # 1. Score salience
    salience = score_salience(text)
    max_score = salience["max_score"]
    print(f"  salience: surprise={salience['surprise']:.2f} cost={salience['cost']:.2f} identity={salience['identity']:.2f} max={max_score:.2f}")

    c = conn.cursor()
    c.execute("UPDATE events SET salience_score=? WHERE id=?", (max_score, eid))
    conn.commit()

    if max_score < SALIENCE_THRESHOLD:
        print("  -> below threshold, skipping ingestion")
        c.execute("UPDATE events SET processed=1 WHERE id=?", (eid,))
        conn.commit()
        return

    # 2. Extract entities
    extracted = extract_entities(text)
    entities = extracted.get("entities", [])
    relations = extracted.get("relations", [])
    print(f"  entities: {len(entities)}, relations: {len(relations)}")

    created_nodes = {}

    # 3. Ingest entities as nodes
    for ent in entities:
        name = ent.get("name", "").strip()
        etype = ent.get("type", "concept")
        if not name:
            continue
        node_type = node_type_from_entity(etype)
        node_id = f"auto:{hashlib.sha256(name.lower().encode()).hexdigest()[:16]}"
        label = name[:50]
        content = f"{name} ({etype}) extracted from event {eid[:8]}"

        # Check if exists
        c.execute("SELECT id FROM nodes WHERE id=?", (node_id,))
        if c.fetchone():
            print(f"  -> node {node_id} exists, updating working memory")
            update_working_memory(conn, node_id)
            created_nodes[name] = node_id
            continue

        # Generate embedding
        emb = embed_text(content)

        # Semantic dedup
        action, dup_id, sim = semantic_dedup(conn, emb, existing_embeddings)
        if action == "merge":
            print(f"  -> merged with {dup_id} (sim={sim:.3f})")
            node_id = dup_id
            update_working_memory(conn, node_id)
            created_nodes[name] = node_id
            continue

        # Create new node
        upsert_node(conn, node_id, node_type, label, content, activation=0.7, source="hippocampus")
        update_node_embedding(conn, node_id, emb)
        existing_embeddings.append((node_id, np.frombuffer(emb, dtype=np.float32)))
        update_working_memory(conn, node_id)
        created_nodes[name] = node_id
        print(f"  -> created node {node_id} ({node_type})")

        if action == "link" and dup_id:
            insert_edge(conn, f"edge:{node_id}:similar_to:{dup_id}", node_id, "similar_to", dup_id, weight=sim, confidence=sim)
            print(f"  -> linked to {dup_id} (sim={sim:.3f})")

    # 4. Ingest relations as edges
    valid_preds = {'caused_by','replaced','contradicts','part_of','similar_to','predicts','committed_to','has_capability','serves','authored','uses','depends_on','predicted_by'}
    for rel in relations:
        subj = rel.get("subject", "").strip()
        pred = rel.get("predicate", "").strip()
        obj = rel.get("object", "").strip()
        if not subj or not obj or pred not in valid_preds:
            continue
        src = created_nodes.get(subj)
        tgt = created_nodes.get(obj)
        if not src or not tgt:
            # Try to find in DB by label similarity
            c.execute("SELECT id FROM nodes WHERE label LIKE ? LIMIT 1", (f"%{subj}%",))
            row = c.fetchone()
            if row:
                src = row[0]
            c.execute("SELECT id FROM nodes WHERE label LIKE ? LIMIT 1", (f"%{obj}%",))
            row = c.fetchone()
            if row:
                tgt = row[0]
        if src and tgt:
            edge_id = f"edge:{src}:{pred}:{tgt}:{uuid.uuid4().hex[:8]}"
            if insert_edge(conn, edge_id, src, pred, tgt, weight=0.8, confidence=0.7):
                print(f"  -> edge {src} --[{pred}]--> {tgt}")

    # 5. Mark event processed
    c.execute("UPDATE events SET processed=1 WHERE id=?", (eid,))
    conn.commit()
    print(f"  -> event {eid[:8]}... processed")


def process_unprocessed_events(conn: sqlite3.Connection, limit: int = MAX_EVENTS_PER_RUN):
    c = conn.cursor()
    c.execute("""
        SELECT id, event_type, payload FROM events
        WHERE processed = 0
        ORDER BY timestamp ASC
        LIMIT ?
    """, (limit,))
    events = c.fetchall()

    if not events:
        print("[HIPPO] No unprocessed events.")
        return 0

    print(f"[HIPPO] Found {len(events)} unprocessed events.")
    existing = get_all_embeddings(conn)
    for ev in events:
        ingest_event(conn, ev, existing)
    return len(events)


def main():
    parser = argparse.ArgumentParser(description="Memory Hippocampus")
    parser.add_argument("--process", action="store_true", help="Process all unprocessed events once")
    parser.add_argument("--daemon", action="store_true", help="Run continuously (sleep 60s)")
    parser.add_argument("--limit", type=int, default=MAX_EVENTS_PER_RUN, help="Max events per cycle")
    args = parser.parse_args()

    if not args.process and not args.daemon:
        print("Use --process or --daemon")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    try:
        if args.process:
            count = process_unprocessed_events(conn, args.limit)
            print(f"\n[HIPPO] Processed {count} events.")
        elif args.daemon:
            print("[HIPPO] Daemon mode. Poll interval 60s.")
            while True:
                count = process_unprocessed_events(conn, args.limit)
                if count:
                    print(f"[HIPPO] Processed {count} events. Sleeping...")
                time.sleep(60)
    except KeyboardInterrupt:
        print("\n[HIPPO] Shutdown.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
