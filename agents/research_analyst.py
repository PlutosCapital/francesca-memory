#!/usr/bin/env python3
"""
Research Analyst Agent — Francesca stack

Company-level fundamentals, SEC filings, income statement analysis,
valuation ratios, and earnings context via OpenBB FMP + SEC providers.
Synthesises an equity research note via DeepSeek (reasoner if off-peak,
standard chat otherwise); falls back to rule-based narrative if the
model call fails. Writes a structured markdown brief to the vault.

Signal-only — no trading advice, advisory outputs only.

Entry: run(symbol: str) or python3 agents/research_analyst.py --symbol AAPL
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

# ── Paths / config ────────────────────────────────────────────────────────────

OPENBB_BASE = "http://127.0.0.1:8000/api/v1"
VAULT_BRIDGE_PATH = Path("/home/boss/vault_bridge")

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [research-analyst] %(message)s"
)
log = logging.getLogger("research-analyst")

# ── Imports: vault + LLM router ───────────────────────────────────────────────

sys.path.insert(0, str(VAULT_BRIDGE_PATH))
from vault_bridge.writer import write_agent_log, update_today  # noqa: E402

sys.path.insert(0, "/home/boss/francesca/agents")
from llm_router import deepseek_heavy, is_deepseek_offpeak  # noqa: E402


# ── OpenBB helper ─────────────────────────────────────────────────────────────

def _fetch(path: str, params: dict, timeout: int = 20) -> Optional[list]:
    """GET an OpenBB endpoint and return the results list, or None on error."""
    try:
        r = requests.get(f"{OPENBB_BASE}{path}", params=params, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            return results if results else None
        log.warning("OpenBB %s returned %d: %s", path, r.status_code, r.text[:120])
    except Exception as exc:
        log.warning("OpenBB %s failed: %s", path, exc)
    return None


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_profile(symbol: str) -> dict:
    rows = _fetch("/equity/profile", {"symbol": symbol, "provider": "fmp"})
    if not rows:
        return {}
    return rows[0]


def fetch_income(symbol: str, limit: int = 4) -> list[dict]:
    rows = _fetch(
        "/equity/fundamental/income",
        {"symbol": symbol, "provider": "fmp", "limit": limit, "period": "annual"},
    )
    return rows or []


def fetch_balance(symbol: str, limit: int = 2) -> list[dict]:
    rows = _fetch(
        "/equity/fundamental/balance",
        {"symbol": symbol, "provider": "fmp", "limit": limit, "period": "annual"},
    )
    return rows or []


def fetch_cash(symbol: str, limit: int = 2) -> list[dict]:
    rows = _fetch(
        "/equity/fundamental/cash",
        {"symbol": symbol, "provider": "fmp", "limit": limit, "period": "annual"},
    )
    return rows or []


def fetch_ratios(symbol: str) -> dict:
    """TTM ratios: margins, returns, P/E, P/B, dividend yield, debt ratios."""
    rows = _fetch(
        "/equity/fundamental/ratios",
        {"symbol": symbol, "provider": "fmp", "limit": 1, "period": "annual"},
    )
    return rows[0] if rows else {}


def fetch_metrics(symbol: str) -> dict:
    """TTM key metrics: EV, EV/EBITDA, ROIC, graham number, FCF yield."""
    rows = _fetch(
        "/equity/fundamental/metrics",
        {"symbol": symbol, "provider": "fmp", "limit": 1, "period": "annual"},
    )
    return rows[0] if rows else {}


def fetch_filings(symbol: str, limit: int = 10) -> list[dict]:
    """SEC filings — 8-K, 10-Q, 10-K, DEF 14A, Form 4, etc."""
    rows = _fetch(
        "/equity/fundamental/filings",
        {"symbol": symbol, "provider": "sec", "limit": limit},
    )
    return rows or []


def fetch_price_target(symbol: str) -> dict:
    rows = _fetch(
        "/equity/estimates/consensus", {"symbol": symbol, "provider": "fmp"}
    )
    return rows[0] if rows else {}


def fetch_forward_estimates(symbol: str, limit: int = 2) -> list[dict]:
    rows = _fetch(
        "/equity/estimates/historical",
        {"symbol": symbol, "provider": "fmp", "limit": limit},
    )
    return rows or []


# ── Derived metrics ───────────────────────────────────────────────────────────

def _pct_change(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
    if curr is None or prev is None or prev == 0:
        return None
    return round((curr - prev) / abs(prev) * 100, 2)


def compute_growth(income: list[dict]) -> dict:
    """Revenue / EPS / net income YoY growth from the two most recent fiscal years."""
    if len(income) < 2:
        return {"revenue_yoy": None, "net_income_yoy": None, "diluted_eps_yoy": None}
    # FMP returns most-recent-first
    latest, prior = income[0], income[1]
    return {
        "revenue_yoy": _pct_change(latest.get("revenue"), prior.get("revenue")),
        "net_income_yoy": _pct_change(
            latest.get("consolidated_net_income") or latest.get("bottom_line_net_income"),
            prior.get("consolidated_net_income") or prior.get("bottom_line_net_income"),
        ),
        "diluted_eps_yoy": _pct_change(
            latest.get("diluted_earnings_per_share"),
            prior.get("diluted_earnings_per_share"),
        ),
    }


def compute_margin_trend(income: list[dict]) -> dict:
    """Gross / operating / net margin, latest and YoY delta (pp)."""
    def margin(row: dict) -> dict:
        rev = row.get("revenue") or 0
        if not rev:
            return {"gross": None, "operating": None, "net": None}
        return {
            "gross": round((row.get("gross_profit") or 0) / rev * 100, 2),
            "operating": round((row.get("total_operating_income") or 0) / rev * 100, 2),
            "net": round(
                (row.get("consolidated_net_income")
                 or row.get("bottom_line_net_income") or 0) / rev * 100,
                2,
            ),
        }

    if not income:
        return {"latest": {}, "delta_pp": {}}
    latest = margin(income[0])
    if len(income) < 2:
        return {"latest": latest, "delta_pp": {}}
    prior = margin(income[1])
    delta = {
        k: (round(latest[k] - prior[k], 2)
            if latest.get(k) is not None and prior.get(k) is not None
            else None)
        for k in ("gross", "operating", "net")
    }
    return {"latest": latest, "delta_pp": delta}


def classify_valuation(ratios: dict, metrics: dict) -> str:
    """
    Rule-based valuation label:
      - P/E <15 AND P/B <3 AND EV/EBITDA <10 → CHEAP
      - P/E 15–25 AND EV/EBITDA 10–15       → FAIR
      - P/E >25 OR EV/EBITDA >20            → RICH
      - else                                 → MIXED
    """
    pe = ratios.get("price_to_earnings")
    pb = ratios.get("price_to_book")
    ev_ebitda = metrics.get("ev_to_ebitda")

    if pe is None and ev_ebitda is None:
        return "UNKNOWN"

    if pe is not None and pb is not None and ev_ebitda is not None:
        if pe < 15 and pb < 3 and ev_ebitda < 10:
            return "CHEAP"
        if 15 <= pe <= 25 and 10 <= ev_ebitda <= 15:
            return "FAIR"
    if (pe is not None and pe > 25) or (ev_ebitda is not None and ev_ebitda > 20):
        return "RICH"
    return "MIXED"


def classify_quality(ratios: dict, metrics: dict) -> str:
    """
    Rule-based quality label based on ROIC, net margin, interest coverage.
      - ROIC >20% AND net margin >15% → HIGH
      - ROIC 10–20% OR net margin 8–15% → MEDIUM
      - ROIC <5% OR negative net margin → LOW
    """
    roic = metrics.get("return_on_invested_capital")
    net_margin = ratios.get("net_profit_margin")
    if roic is None and net_margin is None:
        return "UNKNOWN"
    # OpenBB returns these as decimals (e.g. 0.51 = 51%)
    roic_pct = roic * 100 if roic is not None else None
    nm_pct = net_margin * 100 if net_margin is not None else None

    if roic_pct is not None and roic_pct < 5:
        return "LOW"
    if nm_pct is not None and nm_pct < 0:
        return "LOW"
    if (roic_pct is not None and roic_pct > 20) and (nm_pct is not None and nm_pct > 15):
        return "HIGH"
    return "MEDIUM"


def detect_recent_material_filings(filings: list[dict]) -> list[dict]:
    """Filter for 10-K, 10-Q, 8-K (material), DEF 14A — skip Form 4 insider."""
    material_types = {"10-K", "10-Q", "8-K", "20-F", "DEF 14A", "S-1"}
    out = []
    for f in filings:
        rtype = (f.get("report_type") or "").upper()
        if rtype in material_types:
            out.append(f)
    return out


# ── Narrative synthesis ───────────────────────────────────────────────────────

def build_llm_prompt(
    symbol: str,
    profile: dict,
    income: list[dict],
    growth: dict,
    margin: dict,
    ratios: dict,
    metrics: dict,
    price_target: dict,
    filings: list[dict],
    valuation_label: str,
    quality_label: str,
) -> str:
    latest_inc = income[0] if income else {}
    recent_filings_summary = ", ".join(
        f"{f.get('report_type')} {f.get('filing_date', '')[:10]}"
        for f in filings[:5]
    )

    def _fmt_pct(v):
        return f"{v:+.2f}%" if v is not None else "N/A"

    def _fmt_money(v):
        if v is None:
            return "N/A"
        if abs(v) >= 1e12:
            return f"${v / 1e12:.2f}T"
        if abs(v) >= 1e9:
            return f"${v / 1e9:.2f}B"
        if abs(v) >= 1e6:
            return f"${v / 1e6:.2f}M"
        return f"${v:,.0f}"

    return f"""Company: {profile.get('name', symbol)} ({symbol})
Sector: {profile.get('sector', 'N/A')} / {profile.get('industry_category', 'N/A')}
Market Cap: {_fmt_money(profile.get('market_cap'))}
Last Price: {profile.get('last_price', 'N/A')} {profile.get('currency', '')}

LATEST FISCAL YEAR ({latest_inc.get('fiscal_period', 'N/A')} {latest_inc.get('fiscal_year', 'N/A')} — period ending {latest_inc.get('period_ending', 'N/A')}):
- Revenue: {_fmt_money(latest_inc.get('revenue'))}
- Gross Profit: {_fmt_money(latest_inc.get('gross_profit'))}
- Operating Income: {_fmt_money(latest_inc.get('total_operating_income'))}
- Net Income: {_fmt_money(latest_inc.get('consolidated_net_income') or latest_inc.get('bottom_line_net_income'))}
- Diluted EPS: {latest_inc.get('diluted_earnings_per_share', 'N/A')}

GROWTH (YoY):
- Revenue: {_fmt_pct(growth.get('revenue_yoy'))}
- Net Income: {_fmt_pct(growth.get('net_income_yoy'))}
- Diluted EPS: {_fmt_pct(growth.get('diluted_eps_yoy'))}

MARGINS (latest / YoY Δ in pp):
- Gross: {margin.get('latest', {}).get('gross', 'N/A')}% ({margin.get('delta_pp', {}).get('gross', 'N/A')} pp)
- Operating: {margin.get('latest', {}).get('operating', 'N/A')}% ({margin.get('delta_pp', {}).get('operating', 'N/A')} pp)
- Net: {margin.get('latest', {}).get('net', 'N/A')}% ({margin.get('delta_pp', {}).get('net', 'N/A')} pp)

VALUATION (TTM):
- P/E: {ratios.get('price_to_earnings', 'N/A')}
- P/B: {ratios.get('price_to_book', 'N/A')}
- P/S: {ratios.get('price_to_sales', 'N/A')}
- EV/EBITDA: {metrics.get('ev_to_ebitda', 'N/A')}
- FCF Yield: {ratios.get('price_to_free_cash_flow', 'N/A')} (P/FCF)
- Dividend Yield: {ratios.get('dividend_yield', 'N/A')}

QUALITY (TTM):
- ROIC: {metrics.get('return_on_invested_capital', 'N/A')}
- ROE: {metrics.get('return_on_equity', 'N/A')}
- Net Debt / EBITDA: {metrics.get('net_debt_to_ebitda', 'N/A')}
- Interest Coverage: {ratios.get('interest_coverage_ratio', 'N/A')}

PRICE TARGETS (consensus): high {price_target.get('target_high', 'N/A')} / median {price_target.get('target_median', 'N/A')} / low {price_target.get('target_low', 'N/A')}

RECENT SEC FILINGS: {recent_filings_summary or 'None'}

RULE-BASED LABELS: Valuation={valuation_label}, Quality={quality_label}

Write a 200–250 word equity research note in markdown:
1. One-sentence thesis
2. Growth & Margins (2 bullets)
3. Valuation (1 bullet — anchor on EV/EBITDA and P/E vs. sector norms; note the rule-based label)
4. Quality & Balance Sheet (1 bullet — ROIC, leverage)
5. Risks / Watch Items (2 bullets — reference recent filings if relevant)
6. Signal-only disclaimer as a final italic line

Be concise. Do not recommend buying or selling. Use CVD-safe framing (no red/green).
"""


def synthesize_note(prompt: str, off_peak: bool) -> str:
    system = (
        "You are Francesca's Research Analyst. Produce concise, factual equity "
        "research notes for an advisor audience. Signal-only — never recommend "
        "trades. Use CVD-safe language: prefer deep blue / gold / dark red, "
        "never red/green pairings."
    )
    model = "deepseek-reasoner" if off_peak else "deepseek-chat"
    return deepseek_heavy(
        prompt, system=system, model=model, max_tokens=1200, temperature=0.2
    )


def build_stub_note(
    profile: dict,
    growth: dict,
    margin: dict,
    valuation_label: str,
    quality_label: str,
) -> str:
    """Fallback narrative when LLM call fails — rule-based only."""
    name = profile.get("name", profile.get("symbol", "?"))
    rev_g = growth.get("revenue_yoy")
    ni_g = growth.get("net_income_yoy")
    op_m = margin.get("latest", {}).get("operating")
    rev_g_s = f"{rev_g:+.1f}%" if rev_g is not None else "N/A"
    ni_g_s = f"{ni_g:+.1f}%" if ni_g is not None else "N/A"
    op_m_s = f"{op_m:.1f}%" if op_m is not None else "N/A"

    return (
        f"**Thesis (rule-based stub):** {name} screens as "
        f"**{valuation_label}** valuation / **{quality_label}** quality on TTM data.\n\n"
        f"- Revenue YoY: {rev_g_s} | Net Income YoY: {ni_g_s}\n"
        f"- Operating margin: {op_m_s}\n"
        f"- Valuation rule: {valuation_label}; Quality rule: {quality_label}\n\n"
        f"_LLM synthesis unavailable — rule-based stub only. Signal-only, no trade recommendation._"
    )


# ── Vault output ──────────────────────────────────────────────────────────────

def _fmt_pct(v) -> str:
    return f"{v:+.2f}%" if v is not None else "N/A"


def _fmt_ratio(v) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_money(v) -> str:
    if v is None:
        return "N/A"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "N/A"
    if abs(v) >= 1e12:
        return f"${v / 1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}M"
    return f"${v:,.0f}"


def build_vault_body(
    symbol: str,
    profile: dict,
    income: list[dict],
    growth: dict,
    margin: dict,
    ratios: dict,
    metrics: dict,
    filings: list[dict],
    material_filings: list[dict],
    price_target: dict,
    valuation_label: str,
    quality_label: str,
    narrative: str,
    model_used: str,
) -> str:
    name = profile.get("name", symbol)

    # Income table — most recent 4 years
    inc_rows = []
    for row in income[:4]:
        inc_rows.append(
            f"| {row.get('fiscal_year', '?')} | "
            f"{_fmt_money(row.get('revenue'))} | "
            f"{_fmt_money(row.get('gross_profit'))} | "
            f"{_fmt_money(row.get('total_operating_income'))} | "
            f"{_fmt_money(row.get('consolidated_net_income') or row.get('bottom_line_net_income'))} | "
            f"{row.get('diluted_earnings_per_share', 'N/A')} |"
        )
    if not inc_rows:
        inc_rows = ["| — | N/A | N/A | N/A | N/A | N/A |"]

    # Filings table — most recent 10
    filing_rows = []
    for f in filings[:10]:
        filing_rows.append(
            f"| {f.get('filing_date', '')[:10]} | "
            f"{f.get('report_type', '?')} | "
            f"{f.get('primary_doc_description', '')[:40]} | "
            f"[link]({f.get('filing_detail_url', '')}) |"
        )
    if not filing_rows:
        filing_rows = ["| — | N/A | — | — |"]

    # Material filings sentence
    mat_str = (
        ", ".join(
            f"{m.get('report_type')} {m.get('filing_date', '')[:10]}"
            for m in material_filings[:5]
        )
        or "None in last 10 filings"
    )

    lines = [
        f"## Headline",
        "",
        f"- **Sector:** {profile.get('sector', 'N/A')} / {profile.get('industry_category', 'N/A')}",
        f"- **Market Cap:** {_fmt_money(profile.get('market_cap'))}",
        f"- **Last Price:** {profile.get('last_price', 'N/A')} {profile.get('currency', '')}",
        f"- **Valuation Label (rule-based):** {valuation_label}",
        f"- **Quality Label (rule-based):** {quality_label}",
        f"- **Synthesis Model:** {model_used}",
        "",
        "## Research Note",
        "",
        narrative,
        "",
        "## Income Statement (last 4 FY)",
        "",
        "| FY | Revenue | Gross Profit | Op Income | Net Income | Diluted EPS |",
        "|---|---|---|---|---|---|",
        *inc_rows,
        "",
        "**YoY Growth:** Revenue {rev}, Net Income {ni}, Diluted EPS {eps}".format(
            rev=_fmt_pct(growth.get("revenue_yoy")),
            ni=_fmt_pct(growth.get("net_income_yoy")),
            eps=_fmt_pct(growth.get("diluted_eps_yoy")),
        ),
        "",
        "## Margins (latest FY / YoY Δ)",
        "",
        f"- Gross: {margin.get('latest', {}).get('gross', 'N/A')}% "
        f"(Δ {margin.get('delta_pp', {}).get('gross', 'N/A')} pp)",
        f"- Operating: {margin.get('latest', {}).get('operating', 'N/A')}% "
        f"(Δ {margin.get('delta_pp', {}).get('operating', 'N/A')} pp)",
        f"- Net: {margin.get('latest', {}).get('net', 'N/A')}% "
        f"(Δ {margin.get('delta_pp', {}).get('net', 'N/A')} pp)",
        "",
        "## Valuation & Quality (TTM)",
        "",
        f"- P/E: {_fmt_ratio(ratios.get('price_to_earnings'))}  |  "
        f"P/B: {_fmt_ratio(ratios.get('price_to_book'))}  |  "
        f"P/S: {_fmt_ratio(ratios.get('price_to_sales'))}",
        f"- EV/EBITDA: {_fmt_ratio(metrics.get('ev_to_ebitda'))}  |  "
        f"EV/Sales: {_fmt_ratio(metrics.get('ev_to_sales'))}",
        f"- ROIC: {_fmt_ratio(metrics.get('return_on_invested_capital'))}  |  "
        f"ROE: {_fmt_ratio(metrics.get('return_on_equity'))}",
        f"- Net Debt/EBITDA: {_fmt_ratio(metrics.get('net_debt_to_ebitda'))}  |  "
        f"Interest Coverage: {_fmt_ratio(ratios.get('interest_coverage_ratio'))}",
        f"- Dividend Yield: {_fmt_ratio(ratios.get('dividend_yield'))}  |  "
        f"Payout: {_fmt_ratio(ratios.get('dividend_payout_ratio'))}",
        "",
        "## Analyst Consensus",
        "",
        f"- Price Target: high {_fmt_ratio(price_target.get('target_high'))} / "
        f"median {_fmt_ratio(price_target.get('target_median'))} / "
        f"low {_fmt_ratio(price_target.get('target_low'))}",
        "",
        "## SEC Filings (last 10)",
        "",
        f"**Material filings:** {mat_str}",
        "",
        "| Filed | Type | Description | Link |",
        "|---|---|---|---|",
        *filing_rows,
        "",
        f"*Generated: {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M %Z')} — "
        f"advisory only, no trade execution*",
    ]
    return "\n".join(lines)


def build_today_summary(
    symbol: str,
    profile: dict,
    valuation_label: str,
    quality_label: str,
    growth: dict,
) -> str:
    rev_g = growth.get("revenue_yoy")
    rev_g_s = f"Rev {rev_g:+.1f}% YoY" if rev_g is not None else "Rev N/A"
    return (
        f"{symbol} ({profile.get('name', '?')[:30]}) — "
        f"Val: {valuation_label} | Qual: {quality_label} | {rev_g_s}"
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def run(symbol: str, write_today: bool = True) -> dict:
    """Execute one research analyst cycle for a single symbol."""
    symbol = symbol.upper().strip()
    log.info("Starting research analyst run for %s", symbol)
    now = datetime.now(timezone.utc).astimezone()
    off_peak = is_deepseek_offpeak()

    # 1. Fetch
    profile = fetch_profile(symbol)
    if not profile:
        log.error("Profile fetch failed for %s — aborting", symbol)
        return {
            "symbol": symbol,
            "status": "error",
            "error": "profile_fetch_failed",
            "vault_path": None,
        }

    income = fetch_income(symbol, limit=4)
    balance = fetch_balance(symbol, limit=2)
    cash = fetch_cash(symbol, limit=2)
    ratios = fetch_ratios(symbol)
    metrics = fetch_metrics(symbol)
    filings = fetch_filings(symbol, limit=10)
    price_target = fetch_price_target(symbol)
    forward = fetch_forward_estimates(symbol, limit=2)

    log.info(
        "Fetched: income=%d balance=%d cash=%d ratios=%s metrics=%s filings=%d",
        len(income), len(balance), len(cash),
        bool(ratios), bool(metrics), len(filings),
    )

    if not income:
        log.warning("No income statement data for %s — continuing with partial data", symbol)
    if not filings:
        log.warning("No SEC filings for %s", symbol)

    # 2. Compute signals
    growth = compute_growth(income)
    margin = compute_margin_trend(income)
    valuation_label = classify_valuation(ratios, metrics)
    quality_label = classify_quality(ratios, metrics)
    material_filings = detect_recent_material_filings(filings)

    log.info(
        "Signals: valuation=%s quality=%s rev_yoy=%s ni_yoy=%s",
        valuation_label, quality_label,
        growth.get("revenue_yoy"), growth.get("net_income_yoy"),
    )

    # 3. Synthesize narrative
    prompt = build_llm_prompt(
        symbol, profile, income, growth, margin, ratios, metrics,
        price_target, filings, valuation_label, quality_label,
    )
    model_used = "deepseek-reasoner" if off_peak else "deepseek-chat"
    try:
        narrative = synthesize_note(prompt, off_peak)
        log.info("Synthesis OK (%d chars, model=%s)", len(narrative), model_used)
    except Exception as exc:
        log.warning("DeepSeek synth failed: %s — using rule-based stub", exc)
        narrative = build_stub_note(
            profile, growth, margin, valuation_label, quality_label
        )
        model_used = "rule-based-stub"

    # 4. Vault write
    data_quality = "full" if (income and ratios and metrics and filings) else "partial"
    title = f"Research Note — {symbol} {profile.get('name', '')[:40]} — " \
            f"{now.strftime('%Y-%m-%d %H:%M %Z')}"
    body = build_vault_body(
        symbol, profile, income, growth, margin, ratios, metrics,
        filings, material_filings, price_target,
        valuation_label, quality_label, narrative, model_used,
    )
    extra_meta = {
        "symbol": symbol,
        "company": profile.get("name", "?"),
        "sector": profile.get("sector", "N/A"),
        "valuation_label": valuation_label,
        "quality_label": quality_label,
        "revenue_yoy": growth.get("revenue_yoy") if growth.get("revenue_yoy") is not None else "N/A",
        "net_income_yoy": growth.get("net_income_yoy") if growth.get("net_income_yoy") is not None else "N/A",
        "model_used": model_used,
        "off_peak": off_peak,
        "data_quality": data_quality,
    }

    vault_path = None
    try:
        vault_path = write_agent_log(
            agent="research-analyst",
            title=title,
            body=body,
            tags=[
                "research-analyst",
                "agent-log",
                symbol.lower(),
                valuation_label.lower(),
                quality_label.lower(),
            ],
            extra_meta=extra_meta,
        )
        log.info("Vault write: %s", vault_path)
    except Exception as exc:
        log.error("Vault write failed: %s", exc)

    # 5. today.md (optional)
    if write_today:
        try:
            update_today(
                f"Research Analyst — {symbol}",
                build_today_summary(symbol, profile, valuation_label, quality_label, growth),
            )
        except Exception as exc:
            log.error("today.md update failed: %s", exc)

    return {
        "symbol": symbol,
        "status": "completed",
        "valuation_label": valuation_label,
        "quality_label": quality_label,
        "revenue_yoy": growth.get("revenue_yoy"),
        "net_income_yoy": growth.get("net_income_yoy"),
        "diluted_eps_yoy": growth.get("diluted_eps_yoy"),
        "operating_margin": margin.get("latest", {}).get("operating"),
        "filings_count": len(filings),
        "material_filings_count": len(material_filings),
        "model_used": model_used,
        "off_peak": off_peak,
        "data_quality": data_quality,
        "vault_path": str(vault_path) if vault_path else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research Analyst Agent")
    parser.add_argument(
        "--symbol", required=True,
        help="Ticker symbol to analyze (e.g. AAPL, NVDA)",
    )
    parser.add_argument(
        "--no-today", action="store_true",
        help="Skip updating vault/L1-working/today.md",
    )
    args = parser.parse_args()

    result = run(args.symbol, write_today=not args.no_today)
    print("\n" + "=" * 60)
    print(f"Symbol:            {result['symbol']}")
    print(f"Status:            {result['status']}")
    if result["status"] == "completed":
        print(f"Valuation Label:   {result['valuation_label']}")
        print(f"Quality Label:     {result['quality_label']}")
        print(f"Revenue YoY:       {result.get('revenue_yoy')}")
        print(f"Net Income YoY:    {result.get('net_income_yoy')}")
        print(f"Operating Margin:  {result.get('operating_margin')}")
        print(f"Filings fetched:   {result['filings_count']} "
              f"(material: {result['material_filings_count']})")
        print(f"Model used:        {result['model_used']}")
        print(f"Data quality:      {result['data_quality']}")
        print(f"Vault path:        {result['vault_path']}")
    else:
        print(f"Error:             {result.get('error')}")
