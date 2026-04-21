#!/usr/bin/env python3
"""
Macro Economist Agent — Francesca stack
Fetches structural macro data from OpenBB (yield curve, rates, inflation,
GDP, FOMC calendar), classifies macro regime via Claude reasoning, writes
a macro brief to vault, and sends a daily Telegram pre-market brief.

Entry point: run() — called by Hermes cron or directly via __main__.
"""

import json
import os
import sys
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────

OPENBB_BASE = "http://127.0.0.1:8000/api/v1"
VAULT_BRIDGE_PATH = Path("/home/boss/vault_bridge")
STATE_FILE = Path("/home/boss/.hermes/macro-economist-state.json")
HERMES_ENV = Path("/home/boss/.hermes/.env")

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [macro-economist] %(message)s")
log = logging.getLogger("macro-economist")

# ── vault_bridge import ───────────────────────────────────────────────────────

sys.path.insert(0, str(VAULT_BRIDGE_PATH))
from vault_bridge.writer import write_agent_log, update_today  # noqa: E402
from vault_bridge.reader import read_recent_logs  # noqa: E402


# ── OpenBB helpers ────────────────────────────────────────────────────────────

def _fetch(path: str, params: dict) -> Optional[list]:
    """GET an OpenBB endpoint; return results list or None on error."""
    try:
        r = requests.get(f"{OPENBB_BASE}{path}", params=params, timeout=20)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            return results if results else None
        log.warning("OpenBB %s returned %d: %s", path, r.status_code, r.text[:120])
    except Exception as exc:
        log.warning("OpenBB %s failed: %s", path, exc)
    return None


# ── Data fetch — required sections ────────────────────────────────────────────

def fetch_yield_curve() -> Optional[dict]:
    """Fetch latest treasury rates and compute yield curve spreads.

    Federal Reserve provider returns rates as decimals (0.042 = 4.2%).
    We normalise to percentage on read.
    """
    rows = _fetch("/fixedincome/government/treasury_rates", {
        "provider": "federal_reserve",
        "start_date": "2026-01-01",
        "limit": 5,
    })
    if not rows:
        return None
    row = rows[-1]

    def _pct(v) -> Optional[float]:
        return round(float(v) * 100, 4) if v is not None else None

    tenors = {
        "month_1": _pct(row.get("month_1")),
        "month_3": _pct(row.get("month_3")),
        "month_6": _pct(row.get("month_6")),
        "year_1":  _pct(row.get("year_1")),
        "year_2":  _pct(row.get("year_2")),
        "year_5":  _pct(row.get("year_5")),
        "year_10": _pct(row.get("year_10")),
        "year_30": _pct(row.get("year_30")),
        "date":    row.get("date"),
    }
    y2  = tenors["year_2"]
    y10 = tenors["year_10"]
    m3  = tenors["month_3"]
    y5  = tenors["year_5"]
    y30 = tenors["year_30"]

    # Spreads in percentage points; 100bp = 1.0pp
    spread_2s10s = round(y10 - y2,  4) if y10 is not None and y2  is not None else None
    spread_3m10y = round(y10 - m3,  4) if y10 is not None and m3  is not None else None
    spread_5s30s = round(y30 - y5,  4) if y30 is not None and y5  is not None else None

    # Spec §4.1 shape thresholds (1.0pp = 100bp)
    shape = "UNKNOWN"
    if spread_2s10s is not None:
        if spread_2s10s > 1.0:
            shape = "STEEP"
        elif spread_2s10s > 0.0:
            shape = "NORMAL"
        elif spread_2s10s > -0.25:
            shape = "FLAT"
        else:
            shape = "INVERTED"

    return {
        "tenors": tenors,
        "spread_2s10s": spread_2s10s,
        "spread_3m10y": spread_3m10y,
        "spread_5s30s": spread_5s30s,
        "shape": shape,
    }


def fetch_effr() -> Optional[dict]:
    """Fetch Effective Fed Funds Rate and Fed target range.

    Values from Federal Reserve provider are decimals (0.0364 = 3.64%).
    """
    rows = _fetch("/fixedincome/rate/effr", {
        "provider": "federal_reserve",
        "start_date": "2026-01-01",
        "limit": 10,
    })
    if not rows:
        return None
    row = rows[-1]

    def _pct(v) -> Optional[float]:
        return round(float(v) * 100, 4) if v is not None else None

    return {
        "effr":               _pct(row.get("rate")),
        "target_range_upper": _pct(row.get("target_range_upper")),
        "target_range_lower": _pct(row.get("target_range_lower")),
        "date":               row.get("date"),
    }


def fetch_cpi() -> Optional[dict]:
    """Fetch US CPI from OECD.

    OECD returns YoY growth rate as decimal (0.0239 = 2.39%), not an index.
    We read last 13 months to get the trend direction.
    """
    rows = _fetch("/economy/cpi", {
        "provider": "oecd", "country": "united_states",
        "frequency": "monthly",
        "start_date": "2024-01-01",
        "limit": 15,
    })
    if not rows:
        return None

    values: list[float] = []
    for r in rows:
        raw = r.get("value") or r.get("cpi") or r.get("index_value")
        if raw is not None:
            values.append(float(raw))
    if not values:
        return None

    # Values are already YoY rates as decimals → convert to percentage
    yoy = round(values[-1] * 100, 2)

    # 3-month trend: is inflation accelerating or decelerating?
    trend = "UNKNOWN"
    if len(values) >= 4:
        if values[-1] > values[-4] + 0.0005:
            trend = "RISING"
        elif values[-1] < values[-4] - 0.0005:
            trend = "FALLING"
        else:
            trend = "FLAT"

    # Inflation regime (spec §4.3)
    infl_regime = "UNKNOWN"
    if yoy < 2.0:
        infl_regime = "BELOW_TARGET"
    elif yoy <= 2.5:
        infl_regime = "ON_TARGET"
    elif yoy <= 4.0:
        infl_regime = "ELEVATED"
    elif yoy <= 7.0:
        infl_regime = "HIGH"
    else:
        infl_regime = "RUNAWAY"

    # MoM not computable from YoY series; omit
    return {"yoy": yoy, "mom": None, "trend": trend, "inflation_regime": infl_regime}


def fetch_gdp() -> Optional[dict]:
    """Fetch real GDP quarterly data from OECD and compute QoQ annualised.

    OECD returns absolute USD values (e.g. 25049239700000). QoQ calculation is correct as-is.
    """
    rows = _fetch("/economy/gdp/real", {
        "provider": "oecd", "country": "united_states",
        "frequency": "quarter",
        "start_date": "2022-01-01",
        "limit": 16,
    })
    if not rows:
        return None

    values: list[float] = []
    for r in rows:
        raw = r.get("value") or r.get("gdp") or r.get("real_gdp")
        if raw is not None:
            values.append(float(raw))
    if len(values) < 2:
        return None

    latest = values[-1]
    prev   = values[-2]
    # Spec §4.4: QoQ annualised = ((latest/prev)^4 - 1) * 100
    qoq_ann = round(((latest / prev) ** 4 - 1) * 100, 2) if prev else None

    # 4-quarter average
    trend_avg = None
    if len(values) >= 5:
        qoq_list = []
        for i in range(-4, 0):
            if values[i - 1]:
                qoq_list.append(((values[i] / values[i - 1]) ** 4 - 1) * 100)
        if qoq_list:
            trend_avg = round(sum(qoq_list) / len(qoq_list), 2)

    growth_regime = "UNKNOWN"
    if qoq_ann is not None:
        if qoq_ann >= 2.5:
            growth_regime = "EXPANSION"
        elif qoq_ann >= 0:
            growth_regime = "SLOWING"
        else:
            growth_regime = "CONTRACTION"

    return {"qoq_ann": qoq_ann, "trend_4q_avg": trend_avg, "growth_regime": growth_regime}


# ── Data fetch — enrichment sections ─────────────────────────────────────────

def fetch_cli() -> Optional[dict]:
    """Fetch OECD Composite Leading Indicator for US (index ~100, no scaling needed)."""
    rows = _fetch("/economy/composite_leading_indicator", {
        "provider": "oecd", "country": "united_states",
        "start_date": "2024-01-01", "limit": 12,
    })
    if not rows:
        return None
    values = []
    for r in rows:
        raw = r.get("value") or r.get("cli") or r.get("indicator")
        if raw is not None:
            values.append(float(raw))
    if not values:
        return None

    latest = values[-1]
    trend = "UNKNOWN"
    if len(values) >= 4:
        if latest > values[-4] + 0.1:
            trend = "RISING"
        elif latest < values[-4] - 0.1:
            trend = "FALLING"
        else:
            trend = "FLAT"

    return {"value": round(latest, 2), "trend": trend, "above_100": latest > 100}


def fetch_fomc_documents() -> list:
    """Fetch recent FOMC document references (URLs/metadata only)."""
    return _fetch("/economy/fomc_documents", {"limit": 6}) or []


def fetch_calendar() -> list:
    """Fetch economic calendar for next 14 days via FMP provider."""
    today = datetime.now().strftime("%Y-%m-%d")
    end   = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
    return _fetch("/economy/calendar", {
        "provider": "fmp", "start_date": today, "end_date": end,
    }) or []


def fetch_unemployment() -> Optional[float]:
    """Fetch latest US unemployment rate from OECD.

    OECD returns decimal (0.047 = 4.7%). Normalise to percentage on read.
    """
    rows = _fetch("/economy/unemployment", {
        "provider": "oecd", "country": "united_states",
        "start_date": "2024-01-01", "limit": 12,
    })
    if not rows:
        return None
    row = rows[-1]
    raw = row.get("value") or row.get("rate") or row.get("unemployment_rate")
    return round(float(raw) * 100, 2) if raw is not None else None


def fetch_inflation_expectations() -> Optional[dict]:
    """Fetch 1yr and 10yr inflation expectations from Federal Reserve survey."""
    rows = _fetch("/economy/survey/inflation_expectations", {
        "provider": "federal_reserve", "limit": 6,
    })
    if not rows:
        return None
    row = rows[-1]
    return {
        "one_year": row.get("one_year") or row.get("1_year") or row.get("exp_1y"),
        "ten_year": row.get("ten_year") or row.get("10_year") or row.get("exp_10y"),
        "date":     row.get("date"),
    }


def fetch_ecb_rate() -> Optional[float]:
    """Fetch eurozone rate proxy via OECD (germany; euro_area is not a valid country param).

    Returns percentage (OECD decimal × 100).
    """
    rows = _fetch("/economy/interest_rates", {
        "provider": "oecd", "country": "germany",
        "start_date": "2024-01-01", "limit": 6,
    })
    if not rows:
        return None
    row = rows[-1]
    raw = row.get("value") or row.get("rate") or row.get("interest_rate")
    return round(float(raw) * 100, 4) if raw is not None else None


def fetch_pce() -> Optional[dict]:
    """Fetch PCE deflator; requires FRED key. Falls back gracefully."""
    rows = _fetch("/economy/pce", {"provider": "fred", "limit": 13})
    if not rows:
        rows = _fetch("/economy/fred_series", {"symbol": "PCEPI", "provider": "fred", "limit": 13})
    if not rows:
        return None
    values = [float(r["value"]) for r in rows if r.get("value") is not None]
    if len(values) < 2:
        return None
    yoy = None
    if len(values) >= 13:
        yoy = round((values[-1] / values[-13] - 1) * 100, 2) if values[-13] else None
    return {"latest": values[-1], "yoy": yoy}


# ── Stack context ──────────────────────────────────────────────────────────────

def load_stack_context() -> dict[str, Optional[str]]:
    """Read latest vault logs from Risk Monitor and Market Analyst for context."""
    context: dict[str, Optional[str]] = {"risk_monitor": None, "market_analyst": None}
    for agent_name, key in [("risk-monitor", "risk_monitor"), ("market-analyst", "market_analyst")]:
        try:
            logs = read_recent_logs(agent_name, days=2)
            if logs:
                context[key] = logs[0][:2000]
                log.info("Loaded %s context (%d chars)", agent_name, len(context[key]))
            else:
                log.info("No recent vault logs for %s — proceeding without context", agent_name)
        except Exception as exc:
            log.warning("Context load failed for %s: %s", agent_name, exc)
    return context


# ── Derived metrics ───────────────────────────────────────────────────────────

def compute_policy_stance(effr: Optional[float], cpi_yoy: Optional[float]) -> str:
    """Classify Fed policy stance vs. estimated neutral rate (spec §4.2)."""
    if effr is None or cpi_yoy is None:
        return "UNKNOWN"
    neutral = cpi_yoy + 0.5
    spread  = effr - neutral
    if spread > 1.0:
        return "RESTRICTIVE"
    elif spread > -1.0:
        return "NEUTRAL"
    return "ACCOMMODATIVE"


def infer_policy_regime(prev_effr: Optional[float], current_effr: Optional[float]) -> str:
    """Infer TIGHTENING/PAUSE/EASING from EFFR trend across runs."""
    if prev_effr is None or current_effr is None:
        return "PAUSE"
    delta = current_effr - prev_effr
    if delta > 0.1:
        return "TIGHTENING"
    elif delta < -0.1:
        return "EASING"
    return "PAUSE"


# ── Claude reasoning ──────────────────────────────────────────────────────────

def _load_api_key() -> Optional[str]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    if HERMES_ENV.exists():
        for line in HERMES_ENV.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def classify_regime_with_claude(macro_data: dict, context: dict) -> dict:
    """Call Claude to classify macro regime. Falls back to rule-based if unavailable."""
    try:
        import anthropic
    except ImportError:
        log.warning("anthropic SDK not installed — using rule-based fallback")
        return _rule_based_classification(macro_data)

    api_key = _load_api_key()
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not found — using rule-based fallback")
        return _rule_based_classification(macro_data)

    # Build context snippets from stack logs
    risk_ctx = ""
    if context.get("risk_monitor"):
        fm_lines = [l.strip() for l in context["risk_monitor"].splitlines()
                    if ":" in l and not l.startswith("#") and not l.startswith("---")]
        risk_ctx = "\nRisk Monitor (latest): " + " | ".join(fm_lines[:6])

    market_ctx = ""
    if context.get("market_analyst"):
        fm_lines = [l.strip() for l in context["market_analyst"].splitlines()
                    if ":" in l and not l.startswith("#") and not l.startswith("---")]
        market_ctx = "\nMarket Analyst (latest): " + " | ".join(fm_lines[:6])

    # Strip internal keys before sending to Claude
    clean_data = {k: v for k, v in macro_data.items() if not k.startswith("_")}

    prompt = f"""You are the Macro Economist agent in the Francesca investment stack.
Classify the current US macro regime from the structured data below.

FRAMEWORK:

Quadrant (growth-inflation):
  REFLATION      — growth expanding, inflation rising
  GOLDILOCKS     — growth expanding, inflation falling/stable
  STAGFLATION    — growth contracting, inflation rising
  DEFLATION_RISK — growth contracting, inflation falling

Cycle position (yield curve shape, CLI trend, unemployment):
  EARLY_CYCLE  — yield curve steepening, CLI rising from trough, unemployment falling
  MID_CYCLE    — yield curve normal/flat, CLI >100 stable, unemployment near lows
  LATE_CYCLE   — yield curve flat/inverting, CLI rolling over, unemployment low but rising
  RECESSION    — 2s10s < -25bp, CLI <100 falling, 2+ quarters negative GDP

Policy regime:
  TIGHTENING — rates rising, hawkish Fed guidance
  PAUSE      — rates stable, data-dependent Fed
  EASING     — rates falling, dovish Fed guidance

Composite label examples: GOLDILOCKS, LATE_CYCLE_TIGHTENING, EARLY_EASING,
  STAGFLATION_RISK, RECESSION_WITH_EASING, REFLATION_RECOVERY

MACRO DATA (all rates/spreads in %):
{json.dumps(clean_data, indent=2, default=str)}
{risk_ctx}
{market_ctx}

Respond with valid JSON only — no markdown, no explanation:
{{
  "quadrant": "...",
  "cycle_position": "...",
  "policy_regime": "...",
  "composite_label": "...",
  "primary_risk": "one sentence — the single most important macro risk to the base case",
  "narrative": "2-3 sentences summarising the macro situation",
  "macro_confidence": <integer 0-100>
}}"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        # Strip markdown fences if model wraps response
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        log.info(
            "Claude: %s (confidence=%s)",
            result.get("composite_label"), result.get("macro_confidence"),
        )
        return result
    except Exception as exc:
        log.error("Claude API call failed: %s — using rule-based fallback", exc)
        return _rule_based_classification(macro_data)


def _rule_based_classification(macro_data: dict) -> dict:
    """Deterministic fallback when Claude is unavailable."""
    yc   = macro_data.get("yield_curve") or {}
    cpi  = macro_data.get("cpi") or {}
    gdp  = macro_data.get("gdp") or {}
    cli  = macro_data.get("cli") or {}

    cpi_trend    = cpi.get("trend", "UNKNOWN")
    growth_reg   = gdp.get("growth_regime", "UNKNOWN")
    shape        = yc.get("shape", "UNKNOWN")
    cli_trend    = cli.get("trend", "UNKNOWN")
    spread_2s10s = yc.get("spread_2s10s")

    expanding       = growth_reg in ("EXPANSION", "SLOWING")
    inflation_rising = cpi_trend == "RISING"

    if expanding and inflation_rising:
        quadrant = "REFLATION"
    elif expanding:
        quadrant = "GOLDILOCKS"
    elif inflation_rising:
        quadrant = "STAGFLATION"
    else:
        quadrant = "DEFLATION_RISK"

    if spread_2s10s is not None and spread_2s10s < -0.25 and cli_trend == "FALLING":
        cycle_position = "RECESSION"
    elif shape in ("FLAT", "INVERTED") and cli_trend in ("FALLING", "FLAT"):
        cycle_position = "LATE_CYCLE"
    elif shape == "STEEP" and cli_trend == "RISING":
        cycle_position = "EARLY_CYCLE"
    else:
        cycle_position = "MID_CYCLE"

    policy_regime  = macro_data.get("_policy_regime", "PAUSE")
    composite      = f"{quadrant}_{cycle_position}" if quadrant != "GOLDILOCKS" else "GOLDILOCKS"

    return {
        "quadrant":        quadrant,
        "cycle_position":  cycle_position,
        "policy_regime":   policy_regime,
        "composite_label": composite,
        "primary_risk":    "Rule-based fallback active — Claude API unavailable",
        "narrative": (
            f"Growth: {growth_reg}, Inflation trend: {cpi_trend}, "
            f"Yield curve: {shape}. Deterministic rule-based classification."
        ),
        "macro_confidence": 40,
    }


# ── State ─────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Telegram ──────────────────────────────────────────────────────────────────

def _load_telegram_config() -> tuple[Optional[str], Optional[str]]:
    token   = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_HOME_CHANNEL")
    if token and chat_id:
        return token, chat_id
    if HERMES_ENV.exists():
        for line in HERMES_ENV.read_text().splitlines():
            line = line.strip()
            if line.startswith("TELEGRAM_BOT_TOKEN="):
                token   = line.split("=", 1)[1].strip()
            elif line.startswith("TELEGRAM_HOME_CHANNEL="):
                chat_id = line.split("=", 1)[1].strip()
    return token, chat_id


def send_telegram(message: str) -> bool:
    token, chat_id = _load_telegram_config()
    if not token or not chat_id:
        log.warning("Telegram not configured — skipping")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        if r.status_code == 200:
            log.info("Telegram message sent")
            return True
        log.warning("Telegram send failed: %d %s", r.status_code, r.text[:120])
    except Exception as exc:
        log.warning("Telegram send error: %s", exc)
    return False


# ── Output formatting ─────────────────────────────────────────────────────────

def _fmt(v: Optional[float], decimals: int = 2, suffix: str = "") -> str:
    return f"{v:.{decimals}f}{suffix}" if v is not None else "N/A"


def _fmt_pct(v: Optional[float], decimals: int = 2) -> str:
    return f"{v:.{decimals}f}%" if v is not None else "N/A"


def _arrow(trend: str) -> str:
    return {"RISING": "↗", "FALLING": "↘", "FLAT": "→"}.get(trend, "")


def _bp(v_pct: Optional[float]) -> Optional[float]:
    """Convert percentage-point spread to basis points (e.g. 0.12 → 12.0)."""
    return round(v_pct * 100, 1) if v_pct is not None else None


def build_vault_body(
    now: datetime,
    yc: Optional[dict],
    effr_data: Optional[dict],
    cpi: Optional[dict],
    inflation_exp: Optional[dict],
    gdp: Optional[dict],
    cli: Optional[dict],
    fomc_docs: list,
    calendar: list,
    unemployment: Optional[float],
    ecb_rate: Optional[float],
    pce: Optional[dict],
    regime: dict,
    stack_context: dict,
    data_quality: str,
) -> str:
    ts          = now.strftime("%Y-%m-%d %H:%M %Z")
    composite   = regime.get("composite_label", "UNKNOWN")
    confidence  = regime.get("macro_confidence", 0)
    quadrant    = regime.get("quadrant", "UNKNOWN")
    cycle       = regime.get("cycle_position", "UNKNOWN")
    policy_reg  = regime.get("policy_regime", "PAUSE")
    narrative   = regime.get("narrative", "")
    primary_risk = regime.get("primary_risk", "")

    # ── Yield curve ────────────────────────────────────────────────────────────
    if yc and yc.get("tenors"):
        t = yc["tenors"]
        yc_rows = [
            f"| 1-month  | {_fmt_pct(t.get('month_1'))} |",
            f"| 3-month  | {_fmt_pct(t.get('month_3'))} |",
            f"| 6-month  | {_fmt_pct(t.get('month_6'))} |",
            f"| 1-year   | {_fmt_pct(t.get('year_1'))}  |",
            f"| 2-year   | {_fmt_pct(t.get('year_2'))}  |",
            f"| 5-year   | {_fmt_pct(t.get('year_5'))}  |",
            f"| 10-year  | {_fmt_pct(t.get('year_10'))} |",
            f"| 30-year  | {_fmt_pct(t.get('year_30'))} |",
        ]
        shape        = yc.get("shape", "UNKNOWN")
        bp_2s10s     = _bp(yc.get("spread_2s10s"))
        bp_3m10y     = _bp(yc.get("spread_3m10y"))
        bp_5s30s     = _bp(yc.get("spread_5s30s"))
        spread_sign  = lambda b: f"{b:+.1f}bp" if b is not None else "N/A"
        spread_line  = (
            f"**Shape: {shape}** — "
            f"2s10s: {spread_sign(bp_2s10s)} | "
            f"3m10y: {spread_sign(bp_3m10y)} | "
            f"5s30s: {spread_sign(bp_5s30s)}"
        )
    else:
        yc_rows     = ["| — | N/A |"]
        shape       = "UNKNOWN"
        spread_line = "Yield curve data unavailable"
        bp_2s10s = bp_3m10y = bp_5s30s = None

    # ── Inflation ──────────────────────────────────────────────────────────────
    cpi_yoy      = (cpi or {}).get("yoy")
    cpi_mom      = (cpi or {}).get("mom")
    cpi_trend    = (cpi or {}).get("trend", "UNKNOWN")
    infl_regime  = (cpi or {}).get("inflation_regime", "UNKNOWN")
    exp_1y       = (inflation_exp or {}).get("one_year")
    exp_10y      = (inflation_exp or {}).get("ten_year")
    pce_yoy      = (pce or {}).get("yoy")

    # ── Growth ─────────────────────────────────────────────────────────────────
    gdp_qoq      = (gdp or {}).get("qoq_ann")
    gdp_trend    = (gdp or {}).get("trend_4q_avg")
    growth_reg   = (gdp or {}).get("growth_regime", "UNKNOWN")
    cli_val      = (cli or {}).get("value")
    cli_trend    = (cli or {}).get("trend", "UNKNOWN")
    cli_above    = (cli or {}).get("above_100")
    cli_label    = (
        f"{_fmt(cli_val, 2)} "
        f"({'above 100' if cli_above else 'below 100'}, {cli_trend.lower()} {_arrow(cli_trend)})"
        if cli_val is not None else "N/A"
    )

    # ── Policy ─────────────────────────────────────────────────────────────────
    effr         = (effr_data or {}).get("effr")
    effr_upper   = (effr_data or {}).get("target_range_upper")
    effr_lower   = (effr_data or {}).get("target_range_lower")
    cpi_for_neut = cpi_yoy if cpi_yoy is not None else 2.5
    neutral      = round(cpi_for_neut + 0.5, 2)
    policy_spread_bp = round((effr - neutral) * 100, 0) if effr is not None else None
    policy_stance = compute_policy_stance(effr, cpi_yoy)

    target_str = (
        f"{_fmt(effr_lower, 2)}–{_fmt(effr_upper, 2)}%"
        if effr_lower is not None and effr_upper is not None
        else "N/A"
    )

    # ── FOMC docs ──────────────────────────────────────────────────────────────
    fomc_lines = []
    for doc in fomc_docs[:3]:
        doc_type = (doc.get("type") or doc.get("category") or doc.get("document_type") or "Document")
        doc_date = str(doc.get("date") or doc.get("published") or "")[:10]
        fomc_lines.append(f"- {doc_type} ({doc_date})")
    if not fomc_lines:
        fomc_lines = ["- No FOMC documents available"]

    # ── Calendar ───────────────────────────────────────────────────────────────
    cal_rows = []
    for event in calendar[:15]:
        evt_date    = str(event.get("date") or event.get("event_date") or "?")[:10]
        evt_name    = event.get("event") or event.get("name") or event.get("description") or "?"
        impact      = str(event.get("importance") or event.get("impact") or "—")
        country     = str(event.get("country") or "")
        if "United States" in country or "US" in country or country == "":
            cal_rows.append(f"| {evt_date} | {str(evt_name)[:45]} | {impact} |")
    if not cal_rows:
        cal_rows = ["| — | No events in window | — |"]

    # ── Stack context summary ──────────────────────────────────────────────────
    def _first_meta_line(raw: Optional[str], keywords: list[str]) -> str:
        if not raw:
            return "N/A"
        for line in raw.splitlines():
            lc = line.lower()
            if any(kw in lc for kw in keywords):
                return line.strip().lstrip("- ").strip()
        return "N/A"

    risk_summary   = _first_meta_line(
        stack_context.get("risk_monitor"),
        ["regime:", "risk_score", "risk score"],
    )
    market_summary = _first_meta_line(
        stack_context.get("market_analyst"),
        ["market_character:", "character:"],
    )

    lines = [
        f"## Regime: {composite} (Confidence: {confidence}/100)",
        "",
        f"**Quadrant:** {quadrant}  ",
        f"**Cycle Position:** {cycle}  ",
        f"**Policy Regime:** {policy_reg}  ",
        "",
        "### Narrative",
        "",
        narrative,
        "",
        f"### Macro Confidence: {confidence}/100",
        "",
        f"Data quality: {data_quality}.",
        "",
        "---",
        "",
        "## Yield Curve",
        "",
        "| Tenor | Yield |",
        "|---|---|",
        *yc_rows,
        "",
        spread_line,
        "",
        "---",
        "",
        "## Inflation",
        "",
        "| Metric | Value | Trend |",
        "|---|---|---|",
        f"| CPI YoY          | {_fmt_pct(cpi_yoy)}       | {cpi_trend} |",
        f"| CPI MoM          | {_fmt_pct(cpi_mom, 3)}    | —           |",
        f"| PCE YoY          | {_fmt_pct(pce_yoy)}       | —           |",
        f"| Inflation Regime | {infl_regime}             | —           |",
        f"| 1yr Breakeven    | {_fmt_pct(exp_1y)}        | —           |",
        f"| 10yr Inf. Exp.   | {_fmt_pct(exp_10y)}       | —           |",
        "",
        "---",
        "",
        "## Growth",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Real GDP QoQ (annualised) | {_fmt_pct(gdp_qoq)}   |",
        f"| GDP Trend (4Q avg)        | {_fmt_pct(gdp_trend)} |",
        f"| Growth Regime             | {growth_reg}          |",
        f"| OECD CLI                  | {cli_label}           |",
        f"| Unemployment              | {_fmt_pct(unemployment)} |",
        "",
        "---",
        "",
        "## Policy",
        "",
        "| Central Bank | Rate | Stance |",
        "|---|---|---|",
        f"| Federal Reserve (EFFR) | {_fmt_pct(effr)} | {policy_reg} |",
        f"| Fed Target Range       | {target_str}     | —            |",
        f"| ECB                    | {_fmt_pct(ecb_rate)} | —        |",
        "",
        f"Estimated neutral rate: {_fmt_pct(neutral)}  ",
        f"Policy spread vs neutral: {_fmt(policy_spread_bp, 0)}bp → **{policy_stance}**",
        "",
        "---",
        "",
        "## FOMC / Policy Documents",
        "",
        *fomc_lines,
        "",
        "## Upcoming Macro Events (14 days)",
        "",
        "| Date | Event | Impact |",
        "|---|---|---|",
        *cal_rows,
        "",
        "---",
        "",
        "## Context from Stack",
        "",
        f"- **Risk Monitor** (latest): {risk_summary}",
        f"- **Market Analyst** (latest): {market_summary}",
        "",
        f"*Primary Risk: {primary_risk}*",
        "",
        f"*Fetched: {ts} — advisory only, no trade execution*",
    ]
    return "\n".join(lines)


def build_today_summary(
    regime: dict,
    yc: Optional[dict],
    cpi: Optional[dict],
    gdp: Optional[dict],
    effr_data: Optional[dict],
) -> str:
    composite   = regime.get("composite_label", "UNKNOWN")
    confidence  = regime.get("macro_confidence", 0)
    primary_risk = regime.get("primary_risk", "")

    shape    = (yc or {}).get("shape", "?")
    bp_2s10s = _bp((yc or {}).get("spread_2s10s"))
    cpi_yoy  = (cpi or {}).get("yoy")
    cpi_trend = (cpi or {}).get("trend", "")
    gdp_qoq  = (gdp or {}).get("qoq_ann")
    effr     = (effr_data or {}).get("effr")

    spread_str = f"{bp_2s10s:+.0f}bp" if bp_2s10s is not None else "?"
    cpi_str    = f"{cpi_yoy:.1f}%" if cpi_yoy is not None else "?"
    gdp_str    = f"{gdp_qoq:+.1f}%" if gdp_qoq is not None else "?"
    effr_str   = f"{effr:.2f}%" if effr is not None else "?"

    return (
        f"{composite} ({confidence}%) — "
        f"2s10s {spread_str} {shape} | "
        f"CPI {cpi_str} {_arrow(cpi_trend)} | "
        f"GDP {gdp_str} QoQ | "
        f"EFFR {effr_str} | "
        f"Risk: {primary_risk[:60]}"
    )


def build_telegram_brief(
    now: datetime,
    regime: dict,
    yc: Optional[dict],
    cpi: Optional[dict],
    gdp: Optional[dict],
    effr_data: Optional[dict],
    cli: Optional[dict],
    calendar: list,
) -> str:
    date_str    = now.strftime("%Y-%m-%d")
    composite   = regime.get("composite_label", "UNKNOWN")
    confidence  = regime.get("macro_confidence", 0)
    quadrant    = regime.get("quadrant", "?")
    cycle       = regime.get("cycle_position", "?")
    policy_reg  = regime.get("policy_regime", "?")
    primary_risk = regime.get("primary_risk", "")

    shape    = (yc or {}).get("shape", "?")
    bp_2s10s = _bp((yc or {}).get("spread_2s10s"))
    bp_3m10y = _bp((yc or {}).get("spread_3m10y"))
    cpi_yoy  = (cpi or {}).get("yoy")
    cpi_trend = (cpi or {}).get("trend", "")
    gdp_qoq  = (gdp or {}).get("qoq_ann")
    effr     = (effr_data or {}).get("effr")
    cli_val  = (cli or {}).get("value")
    cli_trend = (cli or {}).get("trend", "")

    s2s10s = f"{bp_2s10s:+.0f}bp" if bp_2s10s is not None else "N/A"
    s3m10y = f"{bp_3m10y:+.0f}bp" if bp_3m10y is not None else "N/A"
    cpi_s  = f"{cpi_yoy:.1f}% YoY" if cpi_yoy is not None else "N/A"
    gdp_s  = f"{gdp_qoq:+.1f}% QoQ" if gdp_qoq is not None else "N/A"
    effr_s = f"{effr:.2f}%" if effr is not None else "N/A"
    cli_s  = f"{cli_val:.1f}" if cli_val is not None else "N/A"

    # Top upcoming high-impact events
    upcoming = []
    for event in calendar[:20]:
        impact  = str(event.get("importance") or event.get("impact") or "").upper()
        country = str(event.get("country") or "")
        if impact in ("HIGH", "3") and ("United States" in country or country == ""):
            evt_date = str(event.get("date") or event.get("event_date") or "?")[:10]
            evt_name = str(event.get("event") or event.get("name") or "?")[:30]
            upcoming.append(f"{evt_date}: {evt_name}")
        if len(upcoming) >= 3:
            break

    events_line = f"\n\nNext: {' | '.join(upcoming)}" if upcoming else ""

    return (
        f"<b>🏦 MACRO BRIEF — {date_str}</b>\n\n"
        f"Regime: <b>{composite}</b> ({confidence}%)\n"
        f"Quadrant: {quadrant} | Cycle: {cycle} | Policy: {policy_reg}\n\n"
        f"Yield Curve: <b>{shape}</b> (2s10s {s2s10s}, 3m10y {s3m10y})\n"
        f"CPI: {cpi_s} {_arrow(cpi_trend)} | EFFR: {effr_s}\n"
        f"GDP: {gdp_s} | CLI: {cli_s} {_arrow(cli_trend)}\n\n"
        f"Primary Risk: {primary_risk[:100]}"
        f"{events_line}"
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def run(send_daily_brief: bool = True) -> dict:
    """
    Execute one macro economist cycle.
    Returns result dict: composite_label, quadrant, cycle_position,
    policy_regime, macro_confidence, vault_path, alerted, data_quality.
    """
    log.info("Starting macro economist run")
    now = datetime.now(timezone.utc).astimezone()

    # 1. Fetch required sections
    required_failures = 0

    log.info("Fetching yield curve...")
    yc = fetch_yield_curve()
    if yc is None:
        log.error("Yield curve fetch failed (required)")
        required_failures += 1

    log.info("Fetching EFFR...")
    effr_data = fetch_effr()
    if effr_data is None:
        log.error("EFFR fetch failed (required)")
        required_failures += 1

    log.info("Fetching CPI...")
    cpi = fetch_cpi()
    if cpi is None:
        log.error("CPI fetch failed (required)")
        required_failures += 1

    log.info("Fetching GDP...")
    gdp = fetch_gdp()
    if gdp is None:
        log.error("GDP fetch failed (required)")
        required_failures += 1

    # 2. Fetch enrichment sections (silent failure per spec §9)
    log.info("Fetching CLI...")
    cli = fetch_cli()

    log.info("Fetching FOMC documents...")
    fomc_docs = fetch_fomc_documents()

    log.info("Fetching calendar...")
    calendar = fetch_calendar()

    log.info("Fetching unemployment...")
    unemployment = fetch_unemployment()

    log.info("Fetching ECB rate...")
    ecb_rate = fetch_ecb_rate()

    log.info("Fetching PCE...")
    pce = fetch_pce()

    log.info("Fetching inflation expectations...")
    inflation_exp = fetch_inflation_expectations()

    # 3. Error stub path: all 4 required sections failed
    if required_failures >= 4:
        log.error("All required data sections failed — writing error stub, skipping Telegram")
        stub_body = (
            f"Run failed: all required data sections unavailable at "
            f"{now.strftime('%Y-%m-%d %H:%M %Z')}. Retry at next cadence."
        )
        vault_path = None
        try:
            vault_path = write_agent_log(
                agent="macro-economist",
                title=f"Macro Brief ERROR — {now.strftime('%Y-%m-%d %H:%M %Z')}",
                body=stub_body,
                tags=["macro-economist", "agent-log", "error"],
                extra_meta={"data_quality": "error", "regime": "UNKNOWN"},
            )
        except Exception as exc:
            log.error("Error stub vault write failed: %s", exc)
        return {
            "composite_label": "UNKNOWN",
            "vault_path": str(vault_path) if vault_path else None,
            "alerted": False,
            "data_quality": "error",
        }

    # 4. Load stack context
    stack_context = load_stack_context()

    # 5. Infer policy regime trend from previous state
    state = load_state()
    prev_effr    = state.get("effr")
    current_effr = (effr_data or {}).get("effr")
    policy_regime = infer_policy_regime(prev_effr, current_effr)

    macro_data = {
        "yield_curve":            yc,
        "effr":                   effr_data,
        "cpi":                    cpi,
        "gdp":                    gdp,
        "cli":                    cli,
        "inflation_expectations": inflation_exp,
        "unemployment":           unemployment,
        "ecb_rate":               ecb_rate,
        "pce":                    pce,
        "_policy_regime":         policy_regime,  # hint for rule-based; stripped before Claude prompt
    }

    # 6. Claude regime classification
    log.info("Classifying regime with Claude...")
    regime = classify_regime_with_claude(macro_data, stack_context)
    # Prefer the inferred policy regime if Claude returns only PAUSE (no rate data to reason on)
    if regime.get("policy_regime") == "PAUSE" and policy_regime != "PAUSE":
        regime["policy_regime"] = policy_regime

    log.info(
        "Regime=%s Quadrant=%s Cycle=%s Policy=%s Confidence=%s",
        regime.get("composite_label"),
        regime.get("quadrant"),
        regime.get("cycle_position"),
        regime.get("policy_regime"),
        regime.get("macro_confidence"),
    )

    # 7. Data quality flag
    data_quality = "full"
    if required_failures > 0:
        data_quality = "partial"
    elif pce is None or inflation_exp is None:
        data_quality = "partial"

    if required_failures >= 2:
        # Floor macro_confidence per spec §9
        regime["macro_confidence"] = min(regime.get("macro_confidence", 30), 30)

    # 8. Build and write vault log
    title = f"Macro Brief — {now.strftime('%Y-%m-%d %H:%M %Z')}"
    body  = build_vault_body(
        now, yc, effr_data, cpi, inflation_exp, gdp, cli,
        fomc_docs, calendar, unemployment, ecb_rate, pce,
        regime, stack_context, data_quality,
    )

    cpi_yoy  = (cpi or {}).get("yoy")
    cpi_trend = (cpi or {}).get("trend", "UNKNOWN")
    gdp_qoq  = (gdp or {}).get("qoq_ann")
    shape    = (yc or {}).get("shape", "UNKNOWN")
    composite = regime.get("composite_label", "UNKNOWN")

    extra_meta = {
        "regime":          composite,
        "quadrant":        regime.get("quadrant", "UNKNOWN"),
        "cycle_position":  regime.get("cycle_position", "UNKNOWN"),
        "policy_regime":   regime.get("policy_regime", "UNKNOWN"),
        "macro_confidence": regime.get("macro_confidence", 0),
        "yield_curve_shape": shape,
        "cpi_yoy":         cpi_yoy if cpi_yoy is not None else "N/A",
        "cpi_trend":       cpi_trend,
        "gdp_qoq_ann":     gdp_qoq if gdp_qoq is not None else "N/A",
        "effr":            current_effr if current_effr is not None else "N/A",
        "primary_risk":    regime.get("primary_risk", ""),
        "data_quality":    data_quality,
    }

    vault_path = None
    try:
        vault_path = write_agent_log(
            agent="macro-economist",
            title=title,
            body=body,
            tags=["macro-economist", "agent-log",
                  composite.lower().replace("_", "-")],
            extra_meta=extra_meta,
        )
        log.info("Vault write: %s", vault_path)
    except Exception as exc:
        log.error("Vault write failed: %s", exc)

    # 9. Update today.md
    try:
        update_today(
            "Macro Economist",
            build_today_summary(regime, yc, cpi, gdp, effr_data),
        )
    except Exception as exc:
        log.error("today.md update failed: %s", exc)

    # 10. Telegram daily brief
    alerted = False
    if send_daily_brief:
        msg = build_telegram_brief(now, regime, yc, cpi, gdp, effr_data, cli, calendar)
        alerted = send_telegram(msg)

    # 11. Persist state
    save_state({
        "composite_label": composite,
        "quadrant":        regime.get("quadrant"),
        "cycle_position":  regime.get("cycle_position"),
        "policy_regime":   regime.get("policy_regime"),
        "macro_confidence": regime.get("macro_confidence"),
        "effr":            current_effr,
        "last_run":        now.strftime("%Y-%m-%d %H:%M %Z"),
        "data_quality":    data_quality,
    })

    return {
        "composite_label": composite,
        "quadrant":        regime.get("quadrant"),
        "cycle_position":  regime.get("cycle_position"),
        "policy_regime":   regime.get("policy_regime"),
        "macro_confidence": regime.get("macro_confidence"),
        "vault_path":      str(vault_path) if vault_path else None,
        "alerted":         alerted,
        "data_quality":    data_quality,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Macro Economist Agent")
    parser.add_argument("--no-telegram", action="store_true", help="Skip Telegram daily brief")
    args = parser.parse_args()
    result = run(send_daily_brief=not args.no_telegram)
    print(f"\n{'='*60}")
    print(f"Regime:    {result['composite_label']}  (Confidence: {result['macro_confidence']})")
    print(f"Quadrant:  {result['quadrant']}")
    print(f"Cycle:     {result['cycle_position']}")
    print(f"Policy:    {result['policy_regime']}")
    print(f"Quality:   {result['data_quality']}")
    print(f"Vault:     {result['vault_path']}")
    print(f"Telegram:  {result['alerted']}")
