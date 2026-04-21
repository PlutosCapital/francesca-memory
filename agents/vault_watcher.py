#!/usr/bin/env python3
"""
Vault Watcher — Phase 1 event emitter.
Polls vault directories and inserts file_write events into memory_graph.db.
No external dependencies beyond stdlib + sqlite3.
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("/home/boss/vault/memory_graph.db")
VAULT = Path("/home/boss/vault")

# Directories to watch
WATCH_PATHS = [
    VAULT / "L1-working" / "today.md",
    VAULT / "L2-episodic" / "daily",
    VAULT / "L2-episodic" / "scribe",
    VAULT / "L2-episodic" / "agent-logs",
]

POLL_INTERVAL = 10  # seconds
MAX_PAYLOAD_BYTES = 2048


def get_file_state() -> dict:
    """Scan all watched paths and return {path: (mtime, size)}."""
    state = {}
    for wp in WATCH_PATHS:
        if wp.is_file():
            stat = wp.stat()
            state[str(wp)] = (stat.st_mtime, stat.st_size)
        elif wp.is_dir():
            for f in wp.rglob("*.md"):
                stat = f.stat()
                state[str(f)] = (stat.st_mtime, stat.st_size)
    return state


def event_id(path: str, mtime: float) -> str:
    return hashlib.sha256(f"{path}:{mtime}".encode()).hexdigest()[:32]


def insert_event(conn: sqlite3.Connection, path: str, mtime: float, size: int):
    """Insert a file_write event. Idempotent by event_id."""
    eid = event_id(path, mtime)
    c = conn.cursor()
    c.execute("SELECT 1 FROM events WHERE id=?", (eid,))
    if c.fetchone():
        return False  # already recorded

    # Read content snippet
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(MAX_PAYLOAD_BYTES)
    except Exception:
        content = ""

    payload = json.dumps({
        "path": path,
        "mtime": mtime,
        "size": size,
        "snippet": content[:1000],
    }, ensure_ascii=False)

    now = datetime.now(timezone.utc).isoformat()
    c.execute("""
        INSERT INTO events (id, timestamp, event_type, payload, salience_score, processed)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (eid, now, "file_write", payload, None, 0))
    conn.commit()
    print(f"[WATCHER] Event {eid[:8]}... {path}")
    return True


def run_once(conn: sqlite3.Connection, previous_state: dict) -> dict:
    """Single poll cycle. Returns new state."""
    current_state = get_file_state()
    for path, (mtime, size) in current_state.items():
        prev = previous_state.get(path)
        if prev is None or prev[0] != mtime:
            insert_event(conn, path, mtime, size)
    return current_state


def run_daemon():
    conn = sqlite3.connect(str(DB_PATH))
    state = get_file_state()  # initial state = baseline, no events
    print(f"[WATCHER] Daemon started. Watching {len(state)} files. Poll={POLL_INTERVAL}s")
    try:
        while True:
            state = run_once(conn, state)
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\n[WATCHER] Shutdown.")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Vault filesystem watcher")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--once", action="store_true", help="Single poll cycle")
    parser.add_argument("--state-file", type=str, default="/tmp/vault_watcher_state.json", help="State file for --once mode")
    args = parser.parse_args()

    if args.once:
        conn = sqlite3.connect(str(DB_PATH))
        previous_state = {}
        if Path(args.state_file).exists():
            with open(args.state_file) as f:
                previous_state = json.load(f)
        previous_state = {k: tuple(v) for k, v in previous_state.items()}
        current_state = run_once(conn, previous_state)
        with open(args.state_file, "w") as f:
            json.dump(current_state, f)
        conn.close()
    elif args.daemon:
        run_daemon()
    else:
        print("Use --daemon or --once")
        sys.exit(1)


if __name__ == "__main__":
    main()
