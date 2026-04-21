# Francesca Memory System

*Personal AI Memory Architecture for Long-Horizon Agents*

---

## What This Is

A dual-layer memory system for a self-hosted AI agent stack. It stores everything the AI learns about its user, its own behavior, and the world — then compresses it automatically so the AI gets smarter over time without manual curation.

Key principle: Memory is not a log file. It is a continuously updated, searchable graph that shapes future decisions.

---

## Architecture Overview


```
┌─────────────────────────────────────────────────────────────┐
│  YOU (Flo) — write notes, chat, make decisions              │
│  ↓ Syncthing syncs to server                                │
├─────────────────────────────────────────────────────────────┤
│  L1 — WORKING MEMORY (today.md)                             │
│  Raw scratchpad. Ephemeral. Lasts 24 hours.                 │
├─────────────────────────────────────────────────────────────┤
│  L2 — EPISODIC MEMORY (agent logs + daily summaries)        │
│  Everything the AI did today, compressed into readable form │
├─────────────────────────────────────────────────────────────┤
│  L3 — SEMANTIC MEMORY (identity + knowledge)                │
│  Long-term facts: who you are, what works, what's decided   │
├─────────────────────────────────────────────────────────────┤
│  L4 — VECTOR SEARCH (embeddings + graph database)           │
│  Ask "what happened last Tuesday?" and get an answer        │
└─────────────────────────────────────────────────────────────┘
```

---

## The Four Layers Explained

### L1 — Working Memory
Where: vault/L1-working/today.md  
What: Your daily scratchpad. Random thoughts, todos, session notes.  
Synced: Via Syncthing to your Mac in real time.  
Lifespan: 24 hours. At midnight, it is archived and compressed into L2.

### L2 — Episodic Memory
Where: vault/L2-episodic/  
What: A time-series record of everything the AI did.

Three sub-layers:
- `agent-logs/` — Raw outputs from every agent (Risk Monitor, Market Analyst, Macro Economist, Prediction Bridge, Morning Brief)
- `scribe/` — Every hour, a cron job reads the latest agent logs and writes a 1-page digest
- `daily/` — Every night at 00:05, a language model (DeepSeek) reads your L1 notes + all agent logs from that day and writes a 400-word summary with five sections: Key Decisions, System Changes, User Patterns, Open Items, Emotional Tone

Deduplication: Before anything gets promoted to long-term memory, a hash check runs. If "Flo likes dark mode" already exists, it doesn't get stored twice — the confidence score just goes up.

### L3 — Semantic Memory
Where: vault/L3-semantic/  
What: Facts that never change (or change slowly).

Files:
- `identity/profile.json` — Who you are, your rules, ADHD profile, color requirements, budget
- `identity/decisions.md` — Active missions, architecture choices, what's pending
- `knowledge/*.json` — Extracted persistent patterns: user preferences, system behavior, strategic decisions

Updated: Sunday at 23:00. The weekly analyzer reads 7 daily summaries, extracts persistent patterns, and updates these files. If a fact appears 3 days in a row, confidence hits 0.9 and it gets promoted from L2 to L3.

### L4 — Vector Search
Where: vault/memory_graph.db (SQLite) + vault/icarus/index/  
What: A searchable database of all L1, L2, and L3 content.

How it works:
- At 01:00 every night, a cron job reads all curated documents
- A local AI model (`nomic-embed-text`, 274MB) converts each document into a 768-number vector
- Stored as a matrix + metadata
- When you search "market regime risk off," it compares your query against every document and returns the top matches ranked by relevance

Search command:

```bash
cd /home/boss/francesca/agents && python3 icarus_indexer.py --search "your question here"
```

---

## The Automation Pipeline

| Time (GMT) | What Happens | Cost |
|------------|--------------|------|
| Every hour | SCRIBE compresses latest agent logs into digest | $0 |
| 00:05 | Daily compressor archives L1, synthesizes L2 summary via DeepSeek | ~$0.003 |
| 01:00 | Icarus rebuilds the search index via Ollama | $0 |
| 07:30 | Prediction bridge fetches Polymarket markets | $0 |
| 08:15 | Morning Brief fires (reads L2 vault files, no API cost) | $0 |
| Sunday 23:00 | Weekly analyzer promotes patterns from L2 → L3 via DeepSeek | ~$0.005 |

Total monthly cost: ~$0.11 for all memory operations.

---

## v1.0 vs v2.0

| | v1.0 (File-Based) | v2.0 (Graph-Based) |
|---|---|---|
| Storage | Markdown + JSON files | SQLite property graph |
| Search | File listing + grep | Semantic vector similarity + graph traversal |
| Deduplication | SHA256 hashes | Embedding cosine similarity |
| Retrieval | Directory walking | 4-stage: working memory → graph neighbors → text search → vector similarity |
| Status | Active — agents write here | Foundation built — parallel system, not yet wired to agents |

v2.0 is a copy, not a replacement. v1.0 files remain untouched. If v2.0 breaks, delete memory_graph.db and v1.0 keeps working.

---

## Why These Choices?

| Decision | Reason |
|----------|--------|
| SQLite, not PostgreSQL/Neo4j | Zero dependencies, single file, transactional, syncs via Syncthing |
| Ollama embeddings, not OpenAI | Free, local, no API keys, no network latency |
| NumPy brute-force, not FAISS | At <10K documents, search is <50ms. No compiled extensions needed |
| Cron + event hybrid, not pure real-time | Balances freshness against cost. Critical events (errors, decisions) will be event-driven in Phase 1 |
| Markdown for humans, graph for machines | Obsidian stays readable. The database is for programmatic retrieval |

---

## How to Query Memory

Search the graph:

```bash
cd /home/boss/francesca/agents
python3 icarus_indexer.py --search "what was the market regime on April 20"
```

Read the latest daily summary:

```bash
cat /home/boss/vault/L2-episodic/daily/$(ls -t /home/boss/vault/L2-episodic/daily/ | head -1)
```

Check deduplicated facts:

```bash
python3 -c "import json; r=json.load(open('/home/boss/vault/L2-episodic/dedup/registry.json')); [print(f'{v['fact'][:60]}... (conf: {v['confidence']})') for v in r.values()]"
```

Verify graph health:

```bash
cd /home/boss/francesca/agents && python3 verify_memory_v2.py
```

---

## For Developers

Schema (SQLite):
- nodes — entities, files, concepts, self-model (24 nodes seeded)
- edges — relationships with temporal validity and confidence (24 edges seeded)
- events — raw ingestions before graph extraction (ready for Phase 1)
- working_memory — hot set for fast retrieval
- node_fts — full-text search index

Embedding model: nomic-embed-text via Ollama (768-dim)

Graph analytics: python-igraph for neighbor traversal and centrality

---

## Roadmap

| Phase | Feature | Status |
|-------|---------|--------|
| 0 | Graph foundation + migration | ✅ Done |
| 1 | Event-driven ingestion (inotify watchers) | Planned |
| 2 | Dynamic context assembly (knapsack selection) | Planned |
| 3 | Active predictions (memory shapes behavior) | Planned |
| 4 | Interference-based decay (forgetting) | Planned |
| 5 | Hermes native integration | Planned |

---

*Built for Francesca — a personal AI operating system. Not affiliated with any employer.*