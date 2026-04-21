#!/usr/bin/env python3
"""
Daily Compressor — L1 + Agent Logs → L2 Daily Summary
NON-DESTRUCTIVE: only archives if today.md has >100 chars
"""

import os
import sys
import glob
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_router import deepseek_heavy, is_deepseek_offpeak

VAULT = Path("/home/boss/vault")
L1 = VAULT / "L1-working"
L2_DAILY = VAULT / "L2-episodic" / "daily"
L1_ARCHIVE = L1 / "archive"
AGENT_LOGS = VAULT / "L2-episodic" / "agent-logs"

for d in [L2_DAILY, L1_ARCHIVE]:
    d.mkdir(parents=True, exist_ok=True)

def read_l1():
    f = L1 / "today.md"
    return f.read_text() if f.exists() else ""

def collect_logs(target_date):
    logs = []
    for d in AGENT_LOGS.glob("*/"):
        for lf in d.glob(f"{target_date}*.md"):
            body = lf.read_text().split("---\n")[-1].strip()[:400]
            logs.append(f"## {d.name}\n{body}")
    return logs

def compress(l1_text, agent_logs, date_str):
    prompt = f"""Synthesize into structured daily memory.

DATE: {date_str}
L1 NOTES: {l1_text[:1500]}
AGENTS: {chr(10).join(agent_logs)[:3000]}

Sections:
1. KEY_DECISIONS
2. SYSTEM_CHANGES
3. USER_PATTERNS
4. OPEN_ITEMS
5. EMOTIONAL_TONE

Rules: Deduplicate. Be specific. CVD-safe colors. Signal-only."""
    
    model = "deepseek-reasoner" if is_deepseek_offpeak() else "deepseek-chat"
    try:
        return deepseek_heavy(prompt, model=model, max_tokens=800, temperature=0.3)
    except Exception as e:
        return f"*Compression failed: {e}*\n\nRaw:\n{l1_text[:500]}"

def run():
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"[COMPRESS] {yesterday}")
    
    l1 = read_l1()
    print(f"[L1] {len(l1)} chars")
    
    logs = collect_logs(yesterday)
    print(f"[AGENTS] {len(logs)} entries")
    
    # Only compress if there's something to compress
    if len(l1) < 50 and not logs:
        print("[SKIP] Nothing to compress")
        return {"status": "skipped", "reason": "empty"}
    
    summary = compress(l1, logs, yesterday)
    print(f"[DEEPSEEK] {len(summary)} chars")
    
    # Write L2
    out = L2_DAILY / f"{yesterday}.md"
    out.write_text(summary)
    print(f"[L2] {out}")
    
    # Archive L1 ONLY if it has content
    if len(l1) > 100:
        archived = L1_ARCHIVE / f"today-{yesterday}.md"
        (L1 / "today.md").rename(archived)
        print(f"[ARCHIVE] {archived}")
        # Create fresh
        (L1 / "today.md").write_text(f"# {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n\n")
        print("[L1] Fresh today.md created")
    else:
        print("[L1] Kept existing (too short to archive)")
    
    return {
        "date": yesterday,
        "l2_path": str(out),
        "l1_chars": len(l1),
        "agent_count": len(logs),
        "summary_chars": len(summary),
        "off_peak": is_deepseek_offpeak(),
    }

if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
