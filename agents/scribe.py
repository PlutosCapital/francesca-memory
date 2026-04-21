#!/usr/bin/env python3
import glob
from datetime import datetime, timezone
from pathlib import Path

VAULT = Path("/home/boss/vault/L2-episodic")
OUT = Path("/home/boss/vault/L2-episodic/scribe")
OUT.mkdir(parents=True, exist_ok=True)

def scribe_hourly():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")
    out = OUT / f"{ts}.md"
    lines = [f"# SCRIBE Hourly Log — {ts}", ""]
    
    # Recurse into all subdirectories for .md files
    for md in sorted(VAULT.rglob("*.md")):
        if "scribe" in str(md):
            continue
        # Only take the latest from each leaf directory
        rel = md.relative_to(VAULT)
        parts = rel.parts
        if len(parts) >= 2:
            agent_name = "/".join(parts[:-1])
        else:
            agent_name = "root"
        # Skip if we already have this agent (take first = latest due to sorted)
        if any(agent_name in l for l in lines):
            continue
        body = md.read_text().split("---\n")[-1].strip()[:200]
        lines += [f"## {agent_name}", body, ""]
    
    out.write_text("\n".join(lines))
    print(f"[SCRIBE] {out}")

if __name__ == "__main__": scribe_hourly()
