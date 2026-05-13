"""Compact decision schema — machine-actionable output for automated trading.

Replaces verbose markdown reports with a dense, score-driven format
designed for AI agents to consume and act on directly.

~500 bytes vs ~5KB markdown. Fits in context windows alongside other signals.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class Conviction(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CONVICTION = "conviction"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TimeHorizon(str, Enum):
    SCALP = "scalp"
    INTRADAY = "intraday"
    SWING = "swing"
    POSITION = "position"


# ── Score components ─────────────────────────────────────────


class SignalScore(BaseModel):
    """Per-analyst score with normalized 0-1 range."""
    score: float = Field(ge=0, le=1, description="Normalized signal strength (0=bearish, 0.5=neutral, 1=bullish)")
    confidence: float = Field(ge=0, le=1, description="Data quality / analyst confidence in this score")
    flags: list[str] = Field(default_factory=list, description="Key flags from this analyst")
    summary: str = Field(default="", max_length=200, description="One-line finding")


class RiskScores(BaseModel):
    """Per-dimension risk assessment."""
    honeypot: float = Field(default=0, ge=0, le=1)
    liquidity: float = Field(default=0, ge=0, le=1)
    concentration: float = Field(default=0, ge=0, le=1)
    mev: float = Field(default=0, ge=0, le=1)
    narrative: float = Field(default=0, ge=0, le=1)
    overall: RiskLevel = RiskLevel.MEDIUM


class ExecutionParams(BaseModel):
    """Machine-actionable execution parameters."""
    entry_price_usd: Optional[float] = None
    position_size_usd: float = 0
    stop_loss_pct: float = 0.15
    take_profit_pct: float = 0.50
    max_slippage_bps: int = 300
    time_horizon: TimeHorizon = TimeHorizon.SWING
    entry_type: str = "limit"
    chain: str = "ethereum"


# ── Compact decision ─────────────────────────────────────────


class CompactDecision(BaseModel):
    """Single compact trading decision — ~500 bytes, machine-actionable."""

    token: str = Field(description="Token address or symbol")
    chain: str = Field(description="Blockchain")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Decision
    action: Action
    conviction: Conviction
    confidence: float = Field(ge=0, le=1, description="Overall decision confidence 0-1")

    # Analyst scores (0=bearish, 0.5=neutral, 1=bullish)
    onchain_score: SignalScore = Field(default_factory=lambda: SignalScore(score=0.5, confidence=0))
    sentiment_score: SignalScore = Field(default_factory=lambda: SignalScore(score=0.5, confidence=0))
    narrative_score: SignalScore = Field(default_factory=lambda: SignalScore(score=0.5, confidence=0))
    technical_score: SignalScore = Field(default_factory=lambda: SignalScore(score=0.5, confidence=0))

    # Execution
    execution: ExecutionParams = Field(default_factory=ExecutionParams)

    # Risk
    risk: RiskScores = Field(default_factory=RiskScores)

    # Thesis (one line only)
    thesis: str = Field(default="", max_length=300, description="Single-line investment thesis")

    @property
    def should_trade(self) -> bool:
        """Gate check: should we execute this? True if BUY + conviction >= MEDIUM + confidence >= 0.5."""
        return (
            self.action == Action.BUY
            and self.conviction in (Conviction.MEDIUM, Conviction.HIGH, Conviction.CONVICTION)
            and self.confidence >= 0.5
            and self.risk.overall != RiskLevel.CRITICAL
        )

    @property
    def signal_summary(self) -> str:
        """Ultra-compact one-line summary for LLM context injection."""
        return (
            f"{self.token[:10]} | {self.action.value.upper()} | "
            f"c:{self.confidence:.0%} | "
            f"onchain:{self.onchain_score.score:.0%} "
            f"sent:{self.sentiment_score.score:.0%} "
            f"narr:{self.narrative_score.score:.0%} "
            f"tech:{self.technical_score.score:.0%} | "
            f"risk:{self.risk.overall.value} | "
            f"${self.execution.position_size_usd:.0f} @ {self.execution.entry_price_usd or 'MKT'}"
        )


# ── Compact batch ─────────────────────────────────────────────


class CompactBatch(BaseModel):
    """Batch of decisions for portfolio-level reasoning."""
    decisions: list[CompactDecision] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    portfolio_cash_usd: float = 0
    portfolio_equity_usd: float = 0
    open_positions: int = 0

    @property
    def actionable(self) -> list[CompactDecision]:
        """Return only decisions that pass the should_trade gate."""
        return [d for d in self.decisions if d.should_trade]

    @property
    def summary(self) -> str:
        """Ultra-compact batch summary."""
        actionable = self.actionable
        lines = [f"Batch: {len(self.decisions)} signals, {len(actionable)} actionable"]
        for d in actionable:
            lines.append(f"  {d.signal_summary}")
        return "\n".join(lines)


# ── Builder from full analysis state ──────────────────────────


def build_compact_decision(state: dict) -> CompactDecision:
    """Extract a CompactDecision from the full CryptoAgent analysis state.

    Parses the structured outputs from Research Manager, Trader, and
    Portfolio Manager to produce a machine-actionable compact decision.
    """
    token = state.get("token_of_interest", "unknown")
    chain = state.get("chain", "ethereum")

    # ── Parse final decision ──
    final = state.get("final_trade_decision", "")
    trader_plan = state.get("trader_investment_plan", "")
    investment_plan = state.get("investment_plan", "")

    # Extract action from Trader's proposal
    action = Action.HOLD
    if "FINAL TRANSACTION PROPOSAL: **BUY**" in trader_plan:
        action = Action.BUY
    elif "FINAL TRANSACTION PROPOSAL: **SELL**" in trader_plan:
        action = Action.SELL

    # Extract conviction from Research Manager
    conviction = Conviction.MEDIUM
    if "High" in investment_plan or "Conviction" in investment_plan:
        conviction = Conviction.HIGH

    # Extract confidence from PM decision
    confidence = 0.6  # default
    if "confiden" in final.lower():
        import re
        m = re.search(r'confiden[ct][ce].*?(\d+)%', final.lower())
        if m:
            confidence = int(m.group(1)) / 100

    # ── Parse analyst reports into scores ──
    onchain_score = _parse_onchain_score(state.get("onchain_report", ""))
    sentiment_score = _parse_sentiment_score(state.get("sentiment_report", ""))
    narrative_score = _parse_narrative_score(state.get("narrative_report", ""))
    technical_score = _parse_technical_score(state.get("technical_report", ""))

    # ── Extract execution params ──
    execution = _parse_execution(trader_plan, chain)

    # ── Extract risk ──
    risk = _parse_risk(final, onchain_score.flags)

    # ── Build thesis ──
    thesis = _extract_thesis(final, investment_plan)

    return CompactDecision(
        token=token,
        chain=chain,
        action=action,
        conviction=conviction,
        confidence=confidence,
        onchain_score=onchain_score,
        sentiment_score=sentiment_score,
        narrative_score=narrative_score,
        technical_score=technical_score,
        execution=execution,
        risk=risk,
        thesis=thesis,
    )


# ── Parsing helpers ────────────────────────────────────────────


def _parse_onchain_score(report: str) -> SignalScore:
    s = 0.5
    c = 0.0
    flags = []

    if not report:
        return SignalScore(score=s, confidence=c)

    report_lower = report.lower()

    # Score from security findings
    if "honeypot" in report_lower:
        if "no" in report_lower or "not a honeypot" in report_lower or "is_honeypot: 0" in report_lower or "false" in report_lower:
            s += 0.15
        else:
            s -= 0.5
            flags.append("honeypot")

    if "ownership renounced" in report_lower or "no owner" in report_lower or "owner: null" in report_lower:
        s += 0.1

    if "proxy" in report_lower and ("no" in report_lower or "not" in report_lower):
        s += 0.05
    elif "proxy" in report_lower:
        s -= 0.1
        flags.append("proxy_upgradeable")

    if "mint" in report_lower and ("no" in report_lower or "not" in report_lower):
        s += 0.05

    # Score from holder distribution
    if "holders" in report_lower:
        import re
        m = re.search(r'(\d[\d,]*)\s*holders', report_lower)
        if m:
            holders = int(m.group(1).replace(",", ""))
            if holders > 50000:
                s += 0.1
            elif holders > 10000:
                s += 0.05
            elif holders < 100:
                s -= 0.1
                flags.append("low_holders")

    # Score from liquidity — find the largest USD amount near 'liquidity'
    if "liquidity" in report_lower:
        import re
        # Find all dollar amounts near 'liquidity'
        amounts = re.findall(r'\$?([\d,]+(?:\.[\d]+)?)\s*(?:usd|USD)?\s*(?:in\s*)?(?:total\s*)?liquidity', report_lower)
        if not amounts:
            # Fallback: any $ amount in the report with 'k' or 'm' suffix near liquidity
            amounts = re.findall(r'liquidity[^$]*\$?([\d,]+(?:\.[\d]+)?)', report_lower)
        if amounts:
            # Take the largest amount found
            liq = max(float(a.replace(",", "")) for a in amounts)
            if liq > 1_000_000:
                s += 0.1
            elif liq > 100_000:
                s += 0.05
            elif liq < 10_000:
                s -= 0.1
                flags.append("low_liquidity")

    # Score from tax
    if "buy tax" in report_lower:
        if "0%" in report_lower or "0.0" in report_lower:
            s += 0.05
        else:
            s -= 0.05
            flags.append("tax_token")

    # Score from age
    if "created" in report_lower or "year" in report_lower:
        if "year" in report_lower:
            s += 0.05
        elif "day" in report_lower:
            s -= 0.05
            flags.append("new_token")

    # Determine confidence based on data quality
    c = 0.7 if "verified" in report_lower or "holders" in report_lower else 0.3

    return SignalScore(
        score=max(0, min(1, s)),
        confidence=max(0, min(1, c)),
        flags=flags,
        summary=_extract_first_sentence(report)[:200],
    )


def _parse_sentiment_score(report: str) -> SignalScore:
    s = 0.5
    c = 0.0
    flags = []

    if not report:
        return SignalScore(score=s, confidence=c)

    report_lower = report.lower()

    # Extract buy/sell ratio
    import re
    m = re.search(r'buy.*?sell.*?ratio.*?(\d+\.?\d*)', report_lower)
    if not m:
        m = re.search(r'ratio.*?(\d+\.?\d*)', report_lower)

    if m:
        ratio = float(m.group(1))
        if ratio > 1.5:
            s += 0.2
        elif ratio > 1.0:
            s += 0.1
        elif ratio > 0.8:
            s -= 0.05
        else:
            s -= 0.15
            flags.append("sell_pressure")

    if "bearish" in report_lower:
        s -= 0.1
    elif "bullish" in report_lower:
        s += 0.1

    c = 0.5

    return SignalScore(
        score=max(0, min(1, s)),
        confidence=c,
        flags=flags,
        summary=_extract_first_sentence(report)[:200],
    )


def _parse_narrative_score(report: str) -> SignalScore:
    s = 0.5
    c = 0.0
    flags = []

    if not report:
        return SignalScore(score=s, confidence=c)

    report_lower = report.lower()

    if "strong alignment" in report_lower:
        s += 0.2
    elif "alignment" in report_lower and "weak" not in report_lower:
        s += 0.1
    elif "misaligned" in report_lower:
        s -= 0.15
        flags.append("narrative_misaligned")

    if "growing" in report_lower or "momentum" in report_lower:
        s += 0.05
    elif "exhausted" in report_lower or "post-peak" in report_lower:
        s -= 0.1
        flags.append("narrative_exhausted")

    c = 0.4

    return SignalScore(
        score=max(0, min(1, s)),
        confidence=c,
        flags=flags,
        summary=_extract_first_sentence(report)[:200],
    )


def _parse_technical_score(report: str) -> SignalScore:
    s = 0.5
    c = 0.0
    flags = []

    if not report:
        return SignalScore(score=s, confidence=c)

    report_lower = report.lower()

    if "bullish" in report_lower:
        s += 0.1
    elif "bearish" in report_lower:
        s -= 0.1
        flags.append("bearish_setup")

    if "downtrend" in report_lower:
        s -= 0.1
    elif "uptrend" in report_lower:
        s += 0.1

    if "overbought" in report_lower:
        s -= 0.05

    c = 0.4

    return SignalScore(
        score=max(0, min(1, s)),
        confidence=c,
        flags=flags,
        summary=_extract_first_sentence(report)[:200],
    )


def _parse_execution(trader_plan: str, chain: str) -> ExecutionParams:
    import re
    params = ExecutionParams(chain=chain)
    if not trader_plan:
        return params

    def _try_float(pattern: str, text: str) -> float | None:
        m = re.search(pattern, text)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                return None
        return None

    # Entry price
    val = _try_float(r'Entry Price.*?([\d]+\.[\d]+)', trader_plan)
    if val:
        params.entry_price_usd = val

    # Position size
    val = _try_float(r'Position Size.*?([\d,]+\.?\d*)', trader_plan)
    if val:
        params.position_size_usd = val

    # Stop loss %
    val = _try_float(r'Stop Loss.*?([\d.]+)\s*%', trader_plan)
    if val:
        params.stop_loss_pct = val / 100

    # Take profit %
    val = _try_float(r'Take Profit.*?([\d.]+)\s*%', trader_plan)
    if val:
        params.take_profit_pct = val / 100

    # Slippage
    val = _try_float(r'Max Slippage.*?([\d.]+)\s*%', trader_plan)
    if val:
        params.max_slippage_bps = int(val * 100)

    # Entry type
    if "Market" in trader_plan:
        params.entry_type = "market"
    elif "Limit" in trader_plan:
        params.entry_type = "limit"
    elif "TWAP" in trader_plan:
        params.entry_type = "twap"

    # Time horizon
    for h in ["scalp", "intraday", "swing", "position", "longterm"]:
        if h in trader_plan.lower():
            try:
                params.time_horizon = TimeHorizon(h if h != "longterm" else "position")
            except ValueError:
                pass
            break

    return params


def _parse_risk(final_decision: str, onchain_flags: list[str]) -> RiskScores:
    risk = RiskScores()
    dl = final_decision.lower()

    if "low liquidity" in dl or "liquidity" in onchain_flags:
        risk.liquidity = 0.6
    if "concentration" in dl or "high concentration" in onchain_flags:
        risk.concentration = 0.5
    if "mev" in dl:
        risk.mev = 0.4
    if "narrative" in dl and ("exhausted" in dl or "risk" in dl):
        risk.narrative = 0.5
    if "honeypot" in onchain_flags:
        risk.honeypot = 0.9
        risk.overall = RiskLevel.CRITICAL
    elif risk.liquidity > 0.5 or risk.concentration > 0.5:
        risk.overall = RiskLevel.HIGH
    elif risk.mev > 0.3 or risk.narrative > 0.3:
        risk.overall = RiskLevel.MEDIUM
    else:
        risk.overall = RiskLevel.LOW

    return risk


def _extract_thesis(final_decision: str, investment_plan: str) -> str:
    """Extract a single-line thesis from the detailed output."""
    # Try the executive summary first
    if "**Executive Summary**" in final_decision:
        idx = final_decision.index("**Executive Summary**")
        rest = final_decision[idx + len("**Executive Summary**"):]
        # Take first meaningful sentence
        rest = rest.strip().lstrip(":").strip()
        sentences = rest.replace("\n", " ").split(".")
        for s in sentences:
            s = s.strip()
            if len(s) > 30:
                return s[:300]

    # Fallback: first line of investment plan
    lines = [l.strip() for l in investment_plan.split("\n") if l.strip() and not l.strip().startswith("**")]
    for l in lines:
        if len(l) > 30:
            return l[:300]

    # Last resort
    lines = [l.strip() for l in final_decision.split("\n") if l.strip()]
    for l in lines:
        if len(l) > 30 and not l.startswith("*") and not l.startswith("#"):
            return l[:300]

    return final_decision[:300]


def _extract_first_sentence(text: str) -> str:
    """Extract first meaningful sentence, skipping preamble."""
    skip_prefixes = [
        "here's", "let me", "i have", "i now", "i'll", "now ",
        "excellent", "great", "alright", "okay", "based on",
    ]
    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 20:
            continue
        if line.startswith("#"):
            continue
        # Skip preamble lines
        lower = line.lower()
        if any(lower.startswith(p) for p in skip_prefixes):
            continue
        if "FINAL TRANSACTION" in line:
            continue
        return line[:200]
    return text[:200]
