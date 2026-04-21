#!/usr/bin/env python3
"""
Deduplication Engine — Shared library for Daily Compressor + Weekly Analyzer.
Normalizes facts, hashes them, manages registry, extracts atomic facts.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

REGISTRY_PATH = Path("/home/boss/vault/L2-episodic/dedup/registry.json")
REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)


def _normalize(text: str) -> str:
    """Strip punctuation, lowercase, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def hash_fact(text: str) -> str:
    """Normalize + SHA256."""
    normalized = _normalize(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_registry() -> dict:
    """Load dedup registry. Returns empty dict if missing."""
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_registry(registry: dict) -> None:
    """Atomic write of registry."""
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    tmp.replace(REGISTRY_PATH)


def check_duplicate(fact_hash: str, registry: Optional[dict] = None) -> Optional[dict]:
    """Return existing entry if confidence > 0.8, else None."""
    if registry is None:
        registry = load_registry()
    entry = registry.get(fact_hash)
    if entry and entry.get("confidence", 0) > 0.8:
        return entry
    return None


def merge_facts(existing: dict, new: dict) -> dict:
    """Bump confidence, update last_confirmed, append sources, increment version."""
    merged = dict(existing)
    # Confidence asymptotically approaches 0.99
    old_conf = existing.get("confidence", 0.5)
    new_conf = new.get("confidence", 0.5)
    merged["confidence"] = min(0.99, max(old_conf, new_conf) + 0.05)
    merged["last_confirmed"] = new.get("last_confirmed", existing.get("last_confirmed"))
    # Deduplicate sources
    sources = set(existing.get("sources", []))
    sources.update(new.get("sources", []))
    merged["sources"] = sorted(sources)
    merged["version"] = existing.get("version", 1) + 1
    # Keep the more specific fact text if new is longer
    if len(new.get("fact", "")) > len(existing.get("fact", "")):
        merged["fact"] = new["fact"]
    # Layer promotion: L2 → L3 if confidence high
    if merged["confidence"] >= 0.9 and merged.get("layer") == "L2":
        merged["layer"] = "L3"
    return merged


def tag_fact_type(sentence: str) -> str:
    """Tag atomic fact as decision, preference, behavior, or unknown."""
    s = sentence.lower()
    if any(w in s for w in ["decided", "decision", "approved", "rejected", "plan", "mission", "will ", "won't "]):
        return "decision"
    if any(w in s for w in ["prefers", "likes", "dislikes", "favorite", "hates", "wants", "doesn't want", "avoid"]):
        return "preference"
    if any(w in s for w in ["works", "breaks", "fails", "bug", "error", "stable", "unstable", "performance", "slow", "fast"]):
        return "behavior"
    if any(w in s for w in ["use ", "using ", "runs on", "deployed", "configured", "installed"]):
        return "infra"
    return "unknown"


def extract_atomic_facts(text: str) -> list[dict]:
    """Split text into sentences, tag each as a fact."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    facts = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 10:
            continue
        facts.append({
            "fact": sent,
            "hash": hash_fact(sent),
            "type": tag_fact_type(sent),
        })
    return facts


def register_facts(facts: list[dict], source: str, layer: str = "L2", date_str: Optional[str] = None) -> dict:
    """Register a list of atomic facts into the dedup registry. Returns updated registry."""
    registry = load_registry()
    if date_str is None:
        from datetime import datetime, timezone
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for f in facts:
        h = f["hash"]
        new_entry = {
            "fact": f["fact"],
            "first_seen": date_str,
            "last_confirmed": date_str,
            "sources": [source],
            "confidence": 0.75,
            "layer": layer,
            "version": 1,
        }
        if h in registry:
            registry[h] = merge_facts(registry[h], new_entry)
        else:
            registry[h] = new_entry
    save_registry(registry)
    return registry


def filter_duplicate_facts(facts: list[dict], threshold: float = 0.8) -> list[dict]:
    """Return only facts that are NOT high-confidence duplicates."""
    registry = load_registry()
    new_facts = []
    for f in facts:
        dup = check_duplicate(f["hash"], registry)
        if not dup:
            new_facts.append(f)
        elif dup.get("confidence", 0) <= threshold:
            # Low confidence duplicate: allow re-registration
            new_facts.append(f)
    return new_facts


if __name__ == "__main__":
    # Self-test
    sample = "Flo prefers dark mode. Flo likes DeepSeek for batch tasks. The system works best with MiniMax for agents."
    facts = extract_atomic_facts(sample)
    print(f"Extracted {len(facts)} facts:")
    for f in facts:
        print(f"  [{f['type']}] {f['fact'][:60]}... (hash: {f['hash'][:16]}...)")
    registry = register_facts(facts, "test")
    print(f"Registry now has {len(registry)} facts.")
