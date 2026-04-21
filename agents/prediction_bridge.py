import os
#!/usr/bin/env python3
"""
Prediction Market Bridge — Polymarket + Kalshi
Zero API keys. Fetches macro-relevant markets, writes probability signals to vault.
"""

import requests
import json
from datetime import datetime, timezone
from pathlib import Path

VAULT_DIR = Path("/home/boss/vault/L2-episodic/agent-logs/prediction-markets")

# ── Polymarket (Gamma API, no auth) ───────────────────────────
def fetch_polymarket():
    url = "https://gamma-api.polymarket.com/events"
    params = {
        "active": "true",
        "closed": "false",
        "limit": 100,
        "order": "volume24hr",
        "ascending": "false",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    events = r.json()

    keywords = ["fed", "rate", "recession", "inflation", "cpi", "gdp", "tariff", "trade",
                "election", "trump", "biden", "war", "ukraine", "israel", "gaza",
                "crypto", "bitcoin", "etf", "nasdaq", "spx", "s&p"]

    now = datetime.now(timezone.utc)
    signals = []

    for ev in events:
        title = (ev.get("title") or "").lower()
        if not any(k in title for k in keywords):
            continue
        for m in ev.get("markets", []):
            try:
                # Skip resolved markets
                end = m.get("endDate")
                if end:
                    try:
                        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
                        if end_dt < now:
                            continue
                    except:
                        pass

                prices = json.loads(m.get("outcomePrices", "[]"))
                if len(prices) < 2:
                    continue

                yes = float(prices[0])
                no = float(prices[1])

                # Skip resolved/certainty extremes
                if yes <= 0.01 or yes >= 0.99:
                    continue

                vol24 = float(m.get("volume24hr", 0))
                if vol24 < 1000:  # Minimum $1k daily volume
                    continue

                signals.append({
                    "source": "polymarket",
                    "question": m.get("question", ev.get("title", "")),
                    "yes_price": yes,
                    "no_price": no,
                    "volume": float(m.get("volume", 0)),
                    "volume24hr": vol24,
                    "liquidity": float(m.get("liquidity", 0)),
                    "expires": end,
                    "slug": m.get("slug", ""),
                })
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return sorted(signals, key=lambda x: x["volume24hr"], reverse=True)

# ── Kalshi (Trade API — authenticated) ────────────────────────
def fetch_kalshi():
    key = os.getenv("KALSHI_API_KEY")
    if not key:
        print("[WARN] KALSHI_API_KEY not set — skipping Kalshi")
        return []

    url = "https://trading-api.kalshi.com/trade-api/v2/markets"
    headers = {"Authorization": key, "Content-Type": "application/json"}
    params = {"limit": 100, "status": "open"}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        if r.status_code == 401:
            print("[WARN] Kalshi 401 — key invalid or expired")
            return []
        r.raise_for_status()
        markets = r.json().get("markets", [])

        keywords = ["fed", "rate", "recession", "inflation", "cpi", "gdp", "tariff",
                    "nfp", "jobs", "unemployment", "election", "spx", "nasdaq"]

        signals = []
        for m in markets:
            t = m.get("title", "").lower()
            if not any(k in t for k in keywords):
                continue
            yes_ask = m.get("yes_ask")
            yes_bid = m.get("yes_bid")
            if yes_ask is not None and yes_bid is not None:
                price = (yes_ask + yes_bid) / 200.0
            elif yes_ask is not None:
                price = yes_ask / 100.0
            else:
                price = 0.5
            signals.append({
                "source": "kalshi",
                "question": m["title"],
                "yes_price": price,
                "no_price": 1.0 - price,
                "volume": m.get("volume", 0),
                "ticker": m.get("ticker"),
                "expires": m.get("expiration_date"),
                "subtitle": m.get("subtitle", "")
            })
        return sorted(signals, key=lambda x: x.get("volume", 0), reverse=True)
    except Exception as e:
        print(f"[WARN] Kalshi fetch failed: {e}")
        return []

# ── Vault Write (JSON + Markdown) ──────────────────────────────
def write_vault(poly, kalshi):
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")

    # JSON
    json_path = VAULT_DIR / f"{ts}.json"
    payload = {
        "timestamp": ts,
        "meta": {
            "poly_count": len(poly),
            "kalshi_count": len(kalshi),
            "poly_volume24hr": sum(s.get("volume24hr", 0) for s in poly),
        },
        "polymarket": poly[:10],
        "kalshi": kalshi[:10],
    }
    json_path.write_text(json.dumps(payload, indent=2))

    # Markdown report (for Morning Brief direct inclusion)
    md_path = VAULT_DIR / f"{ts}.md"
    lines = [
        f"# Prediction Markets — {ts}",
        "",
        f"**Polymarket:** {len(poly)} active macro markets | 24h vol: ${sum(s.get('volume24hr',0) for s in poly):,.0f}",
        f"**Kalshi:** {len(kalshi)} markets (auth required)" if not kalshi else f"**Kalshi:** {len(kalshi)} markets",
        "",
        "## Top Polymarket Signals",
        "",
    ]
    for s in poly[:5]:
        lines.append(f"- **{s['yes_price']:.0%}** — {s['question'][:90]}")
        lines.append(f"  (Vol 24h: ${s['volume24hr']:,.0f} | Exp: {s['expires'][:10] if s['expires'] else 'N/A'})")
        lines.append("")

    if kalshi:
        lines.append("## Top Kalshi Signals")
        lines.append("")
        for s in kalshi[:5]:
            lines.append(f"- **{s['yes_price']:.0%}** — {s['question'][:90]}")
            lines.append(f"  (Vol: ${s.get('volume', 0):,.0f} | Exp: {s.get('expires', 'N/A')})")
            lines.append("")

    md_path.write_text("\n".join(lines))

    print(f"[VAULT] {json_path}")
    print(f"[VAULT] {md_path}")
    return json_path, md_path

# ── Main ───────────────────────────────────────────────────────
def main():
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M')}] Fetching prediction markets...")

    print("[FETCH] Polymarket...")
    poly = fetch_polymarket()
    print(f"  → {len(poly)} active macro markets")

    print("[FETCH] Kalshi...")
    kalshi = fetch_kalshi()
    print(f"  → {len(kalshi)} macro markets")

    j, m = write_vault(poly, kalshi)

    print("\n── Top Polymarket ──")
    for s in poly[:5]:
        print(f"  {s['yes_price']:.0%} | {s['question'][:70]}")

    print(f"\nDone.\nJSON: {j}\nMD:   {m}")

if __name__ == "__main__":
    main()
