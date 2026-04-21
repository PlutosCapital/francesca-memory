#!/usr/bin/env python3
"""
Migration: v1.0 markdown layers → v2.0 SQLite property graph.
Phase 0: schema + data migration only. Idempotent.
"""

import json
import hashlib
import sqlite3
import requests
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path("/home/boss/vault/memory_graph.db")
VAULT = Path("/home/boss/vault")
OLLAMA_URL = "http://127.0.0.1:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


def create_schema(conn: sqlite3.Connection):
    """Create schema if not exists. Idempotent."""
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS nodes (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL CHECK(type IN ('self','user','file','api','decision','error','concept','mission','preference','behavior','prediction')),
        label TEXT NOT NULL,
        content TEXT,
        embedding BLOB,
        activation REAL DEFAULT 0.5 CHECK(activation BETWEEN 0.0 AND 1.0),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        source TEXT DEFAULT 'internal',
        provenance_confidence REAL DEFAULT 1.0,
        is_core BOOLEAN DEFAULT 0,
        token_count INTEGER DEFAULT 0
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS edges (
        id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
        target_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
        predicate TEXT NOT NULL CHECK(predicate IN ('caused_by','replaced','contradicts','part_of','similar_to','predicts','committed_to','has_capability','serves','authored','uses','depends_on','predicted_by')),
        weight REAL DEFAULT 1.0,
        confidence REAL DEFAULT 0.75,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        valid_until TIMESTAMP,
        evidence_count INTEGER DEFAULT 1
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        event_type TEXT CHECK(event_type IN ('file_write','api_error','user_message','agent_output','tool_call','system_change','decision')),
        payload TEXT NOT NULL,
        salience_score REAL,
        processed BOOLEAN DEFAULT 0
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS working_memory (
        node_id TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
        inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        access_count INTEGER DEFAULT 0,
        last_accessed TIMESTAMP
    )
    """)

    # FTS5
    try:
        c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS node_fts USING fts5(content, content_rowid=id, tokenize='porter')")
    except sqlite3.OperationalError:
        pass  # FTS5 may already exist or be unsupported (shouldn't happen)

    # Indexes
    indexes = [
        ("idx_nodes_type", "CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)"),
        ("idx_nodes_activation", "CREATE INDEX IF NOT EXISTS idx_nodes_activation ON nodes(activation)"),
        ("idx_edges_source", "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)"),
        ("idx_edges_target", "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)"),
        ("idx_edges_predicate", "CREATE INDEX IF NOT EXISTS idx_edges_predicate ON edges(predicate)"),
        ("idx_events_processed", "CREATE INDEX IF NOT EXISTS idx_events_processed ON events(processed)"),
    ]
    for name, sql in indexes:
        try:
            c.execute(sql)
        except sqlite3.OperationalError:
            pass

    conn.commit()
    print("[SCHEMA] Created/verified.")


def embed_text(text: str) -> bytes:
    """Generate embedding via Ollama. Returns float32 bytes."""
    payload = text[:512] if text else ""
    r = requests.post(
        OLLAMA_URL,
        json={"model": EMBED_MODEL, "input": payload},
        timeout=60
    )
    r.raise_for_status()
    vec = np.array(r.json()["embeddings"][0], dtype=np.float32)
    return vec.tobytes()


def infer_fact_type(text: str) -> str:
    """Infer node type from fact text."""
    t = text.lower()
    if "key_decisions" in t or "decided" in t or "decision" in t:
        return "decision"
    if "user_patterns" in t or "prefers" in t or "likes" in t or "focused on" in t:
        return "preference"
    if "emotional_tone" in t or "tone is" in t or "satisfaction" in t:
        return "concept"
    if "open_items" in t or "next steps" in t or "potential" in t:
        return "mission"
    if "unavailable" in t or "error" in t or "crash" in t or "failure" in t:
        return "error"
    if "system_changes" in t or "deployed" in t or "implemented" in t or "stack" in t:
        return "behavior"
    return "behavior"  # default


def token_count(text: str) -> int:
    return len(text.split())


def insert_node(c, node_id: str, node_type: str, label: str, content: str,
                activation: float = 0.5, source: str = "internal",
                provenance_confidence: float = 1.0, is_core: bool = False):
    """Insert node with embedding. Idempotent."""
    # Check if exists and has embedding
    c.execute("SELECT embedding FROM nodes WHERE id=?", (node_id,))
    row = c.fetchone()
    if row and row[0] is not None:
        print(f"  [SKIP] {node_id} already exists with embedding.")
        return

    emb = embed_text(content)
    tc = token_count(content)
    now = datetime.now(timezone.utc).isoformat()

    c.execute("""
        INSERT OR REPLACE INTO nodes
        (id, type, label, content, embedding, activation, created_at, updated_at, source, provenance_confidence, is_core, token_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (node_id, node_type, label, content, emb, activation, now, now, source, provenance_confidence, int(is_core), tc))

    # Sync FTS
    c.execute("INSERT OR REPLACE INTO node_fts (rowid, content) VALUES ((SELECT rowid FROM nodes WHERE id=?), ?)", (node_id, content))
    print(f"  [NODE] {node_id} ({node_type})")


def insert_edge(c, edge_id: str, source_id: str, predicate: str, target_id: str,
                weight: float = 1.0, confidence: float = 0.75):
    """Insert edge. Idempotent."""
    c.execute("SELECT 1 FROM edges WHERE id=?", (edge_id,))
    if c.fetchone():
        print(f"  [SKIP] Edge {edge_id} exists.")
        return
    now = datetime.now(timezone.utc).isoformat()
    c.execute("""
        INSERT INTO edges (id, source_id, target_id, predicate, weight, confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (edge_id, source_id, target_id, predicate, weight, confidence, now))
    print(f"  [EDGE] {source_id} --[{predicate}]--> {target_id}")


def migrate():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    create_schema(conn)
    c = conn.cursor()

    print("\n[MIGRATE] Reading v1.0 sources...")

    # --- Self node ---
    profile_path = VAULT / "L3-semantic/identity/profile.json"
    decisions_path = VAULT / "L3-semantic/identity/decisions.md"
    profile_text = profile_path.read_text(encoding="utf-8") if profile_path.exists() else "{}"
    decisions_text = decisions_path.read_text(encoding="utf-8") if decisions_path.exists() else ""
    self_content = f"# Profile\n{profile_text}\n\n# Decisions\n{decisions_text}"

    insert_node(c, "self:francesca", "self", "Francesca", self_content,
                activation=1.0, source="identity", provenance_confidence=1.0, is_core=True)

    # --- User node ---
    insert_node(c, "user:flo", "user", "Flo",
                "Owner of Francesca stack. ADHD profile. CVD-safe colors required.",
                activation=1.0, source="identity", provenance_confidence=1.0, is_core=True)

    # --- Edge: self serves user ---
    insert_edge(c, "edge:self-serves-flo", "self:francesca", "serves", "user:flo",
                weight=1.0, confidence=1.0)

    # --- Core rules as nodes + committed_to edges ---
    try:
        profile = json.loads(profile_text)
        core_rules = profile.get("core_rules", [])
    except json.JSONDecodeError:
        core_rules = []

    for i, rule in enumerate(core_rules):
        rule_id = f"rule:{i+1}"
        insert_node(c, rule_id, "preference", f"Rule {i+1}: {rule[:50]}", rule,
                    activation=1.0, source="identity", provenance_confidence=1.0, is_core=True)
        insert_edge(c, f"edge:self-committed_to-{rule_id}", "self:francesca", "committed_to", rule_id,
                    weight=1.0, confidence=1.0)

    # --- Dedup facts from registry ---
    registry_path = VAULT / "L2-episodic/dedup/registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        for h, entry in registry.items():
            fact_text = entry.get("fact", "")
            fact_type = infer_fact_type(fact_text)
            fact_id = f"fact:{h}"
            insert_node(c, fact_id, fact_type, fact_text[:50], fact_text,
                        activation=entry.get("confidence", 0.75),
                        source=entry.get("sources", ["dedup_engine"])[0],
                        provenance_confidence=entry.get("confidence", 0.75),
                        is_core=False)

    # --- Daily summary ---
    daily_path = VAULT / "L2-episodic/daily/2026-04-20.md"
    daily_content = daily_path.read_text(encoding="utf-8") if daily_path.exists() else ""
    insert_node(c, "daily:2026-04-20", "file", "Daily 2026-04-20", daily_content,
                activation=0.8, source="daily_compressor", provenance_confidence=0.9)

    # --- Scribe logs ---
    scribe_dir = VAULT / "L2-episodic/scribe"
    scribe_files = sorted(scribe_dir.glob("2026-04-21-*.md")) if scribe_dir.exists() else []
    for sf in scribe_files:
        scribe_id = f"scribe:{sf.stem}"
        scribe_content = sf.read_text(encoding="utf-8")
        insert_node(c, scribe_id, "file", f"SCRIBE {sf.stem.split('-')[-1]}", scribe_content,
                    activation=0.6, source="scribe", provenance_confidence=0.8)
        # Link scribe to daily
        insert_edge(c, f"edge:{scribe_id}-part_of-daily", scribe_id, "part_of", "daily:2026-04-20",
                    weight=0.8, confidence=0.7)

    # --- Knowledge files ---
    knowledge_files = [
        ("user_preferences.json", "knowledge:user_preferences", "User Preferences"),
        ("system_behavior.json", "knowledge:system_behavior", "System Behavior"),
        ("strategic_decisions.json", "knowledge:strategic_decisions", "Strategic Decisions"),
    ]
    for filename, node_id, label in knowledge_files:
        kpath = VAULT / "L3-semantic/knowledge" / filename
        kcontent = kpath.read_text(encoding="utf-8") if kpath.exists() else "{}"
        insert_node(c, node_id, "concept", label, kcontent,
                    activation=0.7, source="weekly_analyzer", provenance_confidence=0.8)
        # Link self to knowledge
        insert_edge(c, f"edge:self-part_of-{node_id}", "self:francesca", "part_of", node_id,
                    weight=0.9, confidence=0.8)

    # --- Additional edges ---
    # Daily authored by self
    insert_edge(c, "edge:daily-authored-self", "daily:2026-04-20", "authored", "self:francesca",
                weight=1.0, confidence=1.0)

    # Facts part_of daily
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        for h in registry:
            insert_edge(c, f"edge:fact:{h}-part_of-daily", f"fact:{h}", "part_of", "daily:2026-04-20",
                        weight=0.8, confidence=0.75)

    # Link user to daily (daily is about user's day)
    insert_edge(c, "edge:daily-about-user", "daily:2026-04-20", "authored", "user:flo",
                weight=0.7, confidence=0.8)

    conn.commit()
    conn.close()
    print("\n[MIGRATE] Done.")


if __name__ == "__main__":
    migrate()
