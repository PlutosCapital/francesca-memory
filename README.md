# Francesca Memory System

## What This Repo Contains

| Type | Description |
|------|-------------|
| **Agents** | Python scripts for memory management (compressor, dedup, router, etc.) |
| **Config** | YAML configs for Icarus indexer, Hermes injection, systemd service |
| **Docs** | This file |

---

## What Cannot Be Uploaded (Data Files)

These are **generated data**, not code. They live on the Atlas server and are not version-controlled.

### 1. Binary Files

| File | Size | Description |
|------|------|-------------|
| `vault/memory_graph.db` | 288KB | SQLite property graph database. Contains nodes (entities), edges (relationships), events, working_memory, and FTS5 full-text search index. This is the core v2.0 graph database. |
| `vault/icarus/index/icarus_vectors.pkl` | 31KB | Pickled numpy arrays from Icarus v1.0 vector store. Legacy embeddings for semantic search. |

### 2. L1 — Working Notes (Daily)

| File | Description |
|------|-------------|
| `vault/L1-working/today.md` | Active daily notes. Markdown with frontmatter (date, generated timestamp). Updated throughout the day by Francesca. |
| `vault/L1-working/archive/today-YYYY-MM-DD.md` | Archived daily notes from previous days. |

**Sample `today.md` structure:**
```markdown
---
date: 2026-04-21
generated: 2026-04-21 22:53 GMT
---

# Today

## Test Event
Deployed new risk monitoring dashboard. VIX threshold set to 25.
Flo prefers dark mode for all new dashboards.
```

### 3. L2 — Episodic Logs

| Directory | Description |
|-----------|-------------|
| `vault/L2-episodic/daily/` | Daily summary logs |
| `vault/L2-episodic/scribe/` | Hourly log compaction (scribe agent output) |
| `vault/L2-episodic/dedup/registry.json` | SHA256 dedup registry — tracks which entries are unique |
| `vault/L2-episodic/agent-logs/*/` | Per-agent logs: atlas-audit, francesca, macro-economist, market-analyst, morning-brief, prediction-markets, quant-analyst, research-analyst, risk-monitor |

### 4. L3 — Semantic Knowledge

| Directory | Description |
|-----------|-------------|
| `vault/L3-semantic/identity/` | profile.json, decisions.md, profile.md — Flo's identity & preferences |
| `vault/L3-semantic/knowledge/` | user_preferences.json, system_behavior.json, strategic_decisions.json, raw/ |
| `vault/L3-semantic/knowledge/raw/YYYY-MM-DD.json` | Daily knowledge snapshots |

### 5. Icarus Index (Legacy)

| Directory | Description |
|-----------|-------------|
| `vault/icarus/index/` | icarus_meta.json, icarus_vectors.pkl |
| `vault/icarus/daily/` | Daily vector index dumps |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Francesca Memory                      │
├─────────────────────────────────────────────────────────┤
│  L1-Working    │  L2-Episodic   │  L3-Semantic  │ v2.0  │
│  today.md      │  scribe/       │  identity/    │ Graph │
│  archive/      │  agent-logs/   │  knowledge/   │  DB   │
│                │  daily/        │               │       │
└─────────────────────────────────────────────────────────┘
         ↑                ↑                ↑
    vault_watcher    daily_compressor   memory_
    (Phase 1)        (v1.0)           hippocampus
                                        (Phase 1)
```

---

## Sync Strategy

- **Code** → GitHub (`francesca-memory` repo)
- **Data** → Local filesystem only (Atlas server)
- **Vault** → Syncthing to iMac (~/vault)

This separation keeps the repo lightweight while preserving all memory data locally.

---

## Related Repos

| Repo | Purpose |
|------|---------|
| `hermes-backup-private-` | Full backup of all Python agents (not memory-specific) |
| `francesca-memory` | Memory system code & docs only (this repo) |