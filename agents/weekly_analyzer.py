#!/usr/bin/env python3
"""
Weekly Analyzer — L2 daily summaries → L3 semantic knowledge + identity updates.
Runs Sundays at 23:00 GMT. DeepSeek off-peak preferred. Fallback: Ollama → raw.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dedup_engine import load_registry, register_facts, extract_atomic_facts
from llm_router import deepseek_heavy, ollama_local

VAULT = Path("/home/boss/vault")
L2_DAILY = VAULT / "L2-episodic" / "daily"
L3_IDENTITY = VAULT / "L3-semantic" / "identity"
L3_KNOWLEDGE = VAULT / "L3-semantic" / "knowledge"
L3_KNOWLEDGE_RAW = L3_KNOWLEDGE / "raw"

for d in [L3_KNOWLEDGE, L3_KNOWLEDGE_RAW]:
    d.mkdir(parents=True, exist_ok=True)

CVD_SAFE_INSTRUCTION = (
    "Use CVD-safe colors only: deep blue (#000080), gold (#D4AF37), dark red (#8B0000), black. "
    "Never use red/green traffic light colors."
)

ANALYSIS_PROMPT = """You are Francesca's weekly memory analyst.
Analyze the following 7 daily summaries and extract persistent patterns.

Output MUST be valid JSON with this exact structure:
{{
  "user_preferences": [
    {{"fact": "...", "confidence": 0.92, "evidence": "day1, day3"}}
  ],
  "system_behavior": [
    {{"fact": "...", "confidence": 0.88, "evidence": "day2"}}
  ],
  "strategic_decisions": [
    {{"fact": "...", "confidence": 0.95, "evidence": "day4, day5", "status": "approved|rejected|pending"}}
  ],
  "contradictions": [
    {{"facts": ["...", "..."], "severity": "high|medium|low", "note": "..."}}
  ],
  "profile_updates": {{
    "key": "value"
  }},
  "summary": "One-paragraph executive summary of the week."
}}

Rules:
- Only include facts with confidence >= 0.7.
- Be specific. Include numbers, names, exact configs.
- If a pattern is not persistent (only seen once), omit it or lower confidence.
- {cvd_safe}
- Do NOT wrap in markdown code blocks. Output raw JSON only.

Daily summaries:
---
{content}
---
"""

OLLAMA_FALLBACK_PROMPT = """Analyze these daily summaries. Extract:
1. USER_PREFERENCES (what Flo consistently likes/dislikes)
2. SYSTEM_BEHAVIOR (what works/breaks)
3. STRATEGIC_DECISIONS (approved/rejected plans)
4. CONTRADICTIONS (disagreements between days)

Format as simple bullet points. {cvd_safe}

{content}
"""


def get_last_7_days(ref_date: datetime = None) -> list[str]:
    d = ref_date or datetime.now(timezone.utc)
    dates = []
    for i in range(1, 8):
        dates.append((d - timedelta(days=i)).strftime("%Y-%m-%d"))
    return dates


def read_daily_summaries(dates: list[str]) -> dict[str, str]:
    out = {}
    for ds in dates:
        p = L2_DAILY / f"{ds}.md"
        if p.exists():
            out[ds] = p.read_text(encoding="utf-8")
    return out


def analyze_with_llm(content: str) -> str:
    """Run analysis with fallback chain."""
    prompt = ANALYSIS_PROMPT.format(content=content[:15000], cvd_safe=CVD_SAFE_INSTRUCTION)
    try:
        print("[ANALYZE] Trying DeepSeek reasoner...")
        return deepseek_heavy(
            prompt,
            system="You are a pattern extraction engine. Output valid JSON only. No markdown fences.",
            model="deepseek-reasoner",
            max_tokens=1200,
            temperature=0.2,
        )
    except Exception as e:
        print(f"[WARN] DeepSeek failed: {e}")

    try:
        print("[ANALYZE] Trying Ollama local...")
        ollama_prompt = OLLAMA_FALLBACK_PROMPT.format(content=content[:8000], cvd_safe=CVD_SAFE_INSTRUCTION)
        return ollama_local(ollama_prompt, model=os.getenv("OLLAMA_MODEL", "francesca-local"))
    except Exception as e:
        print(f"[WARN] Ollama failed: {e}")

    print("[ANALYZE] All LLMs failed. Preserving raw concatenation.")
    return f"RAW_PRESERVATION\n\n{content[:4000]}"


def parse_analysis(text: str) -> dict:
    """Extract JSON from model output. Handles markdown fences."""
    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object inside text
        m = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return {}


def load_json_atomic(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_json_atomic(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def merge_json_knowledge(existing: dict, new_items: list[dict], week_str: str) -> dict:
    """Merge extracted items into versioned knowledge JSON."""
    if "facts" not in existing:
        existing["facts"] = []
    if "version" not in existing:
        existing["version"] = 1
    if "last_updated" not in existing:
        existing["last_updated"] = week_str

    existing_facts = {f["fact"]: f for f in existing["facts"]}

    for item in new_items:
        fact_text = item.get("fact", "").strip()
        if not fact_text:
            continue
        conf = item.get("confidence", 0.75)
        if conf < 0.7:
            continue
        if fact_text in existing_facts:
            old = existing_facts[fact_text]
            old["confidence"] = min(0.99, max(old.get("confidence", 0.5), conf) + 0.03)
            old["last_seen"] = week_str
            old["count"] = old.get("count", 1) + 1
        else:
            existing_facts[fact_text] = {
                "fact": fact_text,
                "confidence": conf,
                "first_seen": week_str,
                "last_seen": week_str,
                "count": 1,
                "evidence": item.get("evidence", ""),
                "status": item.get("status", "active"),
            }

    existing["facts"] = list(existing_facts.values())
    existing["version"] += 1
    existing["last_updated"] = week_str
    return existing


def update_profile_json(updates: dict, week_str: str) -> dict:
    profile_path = L3_IDENTITY / "profile.json"
    profile = load_json_atomic(profile_path)
    if not profile:
        profile = {
            "name": "Francesca",
            "owner": "Flo",
            "cvd_safe": True,
            "memory_layers": {"L1": "Working notes", "L2": "Agent logs + scribe", "L3": "Identity + knowledge", "L4": "Icarus vector index"},
        }
    # Apply shallow updates
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(profile.get(k), dict):
            profile[k].update(v)
        else:
            profile[k] = v
    profile["last_analyzed"] = week_str
    save_json_atomic(profile_path, profile)
    return profile


def regenerate_profile_md(profile: dict):
    """Generate human-readable profile.md from profile.json."""
    lines = ["# Francesca Identity Profile", ""]
    lines.append(f"**Name:** {profile.get('name', 'Francesca')}")
    lines.append(f"**Owner:** {profile.get('owner', 'Flo')}")
    lines.append(f"**CVD Safe:** {'Yes' if profile.get('cvd_safe') else 'No'}")
    lines.append(f"**Last Analyzed:** {profile.get('last_analyzed', 'N/A')}")
    lines.append("")

    if "core_rules" in profile:
        lines.append("## Core Rules")
        for rule in profile["core_rules"]:
            lines.append(f"- {rule}")
        lines.append("")

    if "routing" in profile:
        lines.append(f"**Routing:** {profile['routing']}")
    if "budget" in profile:
        b = profile["budget"]
        lines.append(f"**Budget:** {b.get('monthly_total', '?')} {b.get('currency', 'USD')}/month")
    if "infrastructure" in profile:
        lines.append("## Infrastructure")
        for k, v in profile["infrastructure"].items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    if "memory_layers" in profile:
        lines.append("## Memory Layers")
        for k, v in profile["memory_layers"].items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    md_path = L3_IDENTITY / "profile.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[PROFILE_MD] {md_path}")


def append_decisions_delta(week_str: str, summary: str, contradictions: list[dict]):
    decisions_path = L3_IDENTITY / "decisions.md"
    lines = []
    if decisions_path.exists():
        text = decisions_path.read_text(encoding="utf-8")
        # Skip if delta for this week already exists
        if f"## Weekly Delta — {week_str}" in text:
            print(f"[DECISIONS] Delta for {week_str} already exists. Skipping.")
            return
        lines = text.splitlines()

    delta = [
        "",
        f"## Weekly Delta — {week_str}",
        "",
        f"**Summary:** {summary}",
        "",
    ]
    if contradictions:
        delta.append("**Contradictions Flagged:**")
        for c in contradictions:
            delta.append(f"- [{c.get('severity', 'medium').upper()}] {c.get('note', 'Unspecified')}")
        delta.append("")

    decisions_path.write_text("\n".join(lines + delta), encoding="utf-8")
    print(f"[DECISIONS] Appended delta to {decisions_path}")


def run(dry_run: bool = False, force_week: str = None):
    now = datetime.now(timezone.utc)
    week_str = force_week or now.strftime("%Y-%m-%d")
    print(f"[WEEKLY_ANALYZER] Week ending {week_str} (dry_run={dry_run})")

    dates = get_last_7_days(now)
    summaries = read_daily_summaries(dates)
    print(f"[INPUTS] Found {len(summaries)} daily summaries out of 7 days")

    if not summaries:
        print("[INFO] No daily summaries found. Nothing to analyze.")
        return

    content_block = "\n\n---DAY_BREAK---\n\n".join(
        f"# {ds}\n{text}" for ds, text in sorted(summaries.items())
    )

    if dry_run:
        print("[DRY-RUN] Would analyze content:")
        print(content_block[:500] + "...")
        return

    raw_output = analyze_with_llm(content_block)

    # Parse or preserve
    parsed = parse_analysis(raw_output)
    if not parsed and "RAW_PRESERVATION" not in raw_output:
        # Save raw for manual fix
        raw_path = L3_KNOWLEDGE_RAW / f"{week_str}.json"
        raw_path.write_text(json.dumps({"raw": raw_output}, indent=2), encoding="utf-8")
        print(f"[WARN] JSON parse failed. Raw preserved at {raw_path}")
        parsed = {}

    # Merge into L3 knowledge files
    for key, filename in [
        ("user_preferences", "user_preferences.json"),
        ("system_behavior", "system_behavior.json"),
        ("strategic_decisions", "strategic_decisions.json"),
    ]:
        items = parsed.get(key, [])
        if items:
            path = L3_KNOWLEDGE / filename
            existing = load_json_atomic(path)
            merged = merge_json_knowledge(existing, items, week_str)
            save_json_atomic(path, merged)
            print(f"[L3] Updated {filename} ({len(merged['facts'])} facts)")

    # Update profile (always bump last_analyzed)
    profile_updates = parsed.get("profile_updates", {})
    profile = update_profile_json(profile_updates, week_str)
    regenerate_profile_md(profile)
    print(f"[L3] Updated profile.json (last_analyzed={week_str})")

    # Append decisions delta
    summary_text = parsed.get("summary", "No summary generated.")
    contradictions = parsed.get("contradictions", [])
    append_decisions_delta(week_str, summary_text, contradictions)

    # Register high-confidence facts into dedup registry
    all_facts = []
    for key in ["user_preferences", "system_behavior", "strategic_decisions"]:
        for item in parsed.get(key, []):
            if item.get("confidence", 0) >= 0.8:
                all_facts.append({"fact": item["fact"], "hash": "", "type": key.replace("_", "/")})
    # Re-hash
    from dedup_engine import hash_fact
    for f in all_facts:
        f["hash"] = hash_fact(f["fact"])
    if all_facts:
        register_facts(all_facts, source="weekly_analyzer", layer="L3", date_str=week_str)
        print(f"[REGISTRY] Registered {len(all_facts)} L3 facts")

    print("[DONE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly Memory Analyzer")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files or call APIs")
    parser.add_argument("--force-week", type=str, help="Override week date (YYYY-MM-DD)")
    parser.add_argument("--prefer-ollama", action="store_true", help="Force Ollama (testing)")
    args = parser.parse_args()

    if args.prefer_ollama:
        def _only_ollama(prompt, system=None, model="deepseek-reasoner", max_tokens=4000, temperature=0.2):
            return ollama_local(prompt[:8000])
        import llm_router
        llm_router.deepseek_heavy = _only_ollama

    run(dry_run=args.dry_run, force_week=args.force_week)
