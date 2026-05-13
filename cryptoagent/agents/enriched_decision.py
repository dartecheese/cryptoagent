"""Enriched trading decision — intelligent, data-rich, bridge-ready.

Extends CompactDecision with comparative analysis, position sizing rationale,
momentum decomposition, narrative lifecycle, and execution quality prediction.
Designed to feed directly into AST's SafetyBreaker → Actuary → Slinger pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from cryptoagent.agents.compact_decision import (
    Action, Conviction, RiskLevel, TimeHorizon,
    SignalScore, RiskScores, ExecutionParams, CompactDecision,
)


# ── Enriched dimensions ───────────────────────────────────────


class NarrativeLifecycle(BaseModel):
    """Where this token sits in the narrative hype cycle."""
    phase: str = Field(default="unknown", description="emerging | accelerating | peak | declining | dead")
    strength: float = Field(default=0.5, ge=0, le=1, description="Narrative strength/attention score")
    competitors: list[str] = Field(default_factory=list, description="Top 3 competing tokens in same narrative")
    rotation: str = Field(default="neutral", description="inflow | neutral | outflow")


class MomentumDecomposition(BaseModel):
    """Multi-timeframe momentum analysis."""
    scalp_1h: float = Field(default=0, description="1h momentum (-1 to +1)")
    intraday_6h: float = Field(default=0, description="6h momentum")
    swing_24h: float = Field(default=0, description="24h momentum")
    trend_7d: float = Field(default=0, description="7d momentum")
    convergence: str = Field(default="neutral", description="aligning | diverging | neutral")
    volume_trend: str = Field(default="steady", description="rising | steady | falling")


class PositionSizing(BaseModel):
    """Kelly-inspired position sizing with rationale."""
    kelly_fraction: float = Field(default=0, ge=0, le=1, description="Optimal Kelly fraction")
    adjusted_size_usd: float = Field(default=0, description="Size after risk adjustments")
    max_size_usd: float = Field(default=0, description="Hard cap from liquidity constraints")
    rationale: str = Field(default="", description="Why this size")


class ExecutionQuality(BaseModel):
    """Predicted execution quality."""
    estimated_fill_pct: float = Field(default=1.0, ge=0, le=1, description="Likelihood of fill at limit")
    estimated_slippage_bps: int = Field(default=0, description="Expected slippage in bps")
    mev_risk: float = Field(default=0, ge=0, le=1, description="Sandwich attack probability")
    gas_estimate_usd: float = Field(default=0, description="Estimated gas cost")
    recommended_mempool: str = Field(default="public", description="public | private | flashbots")


class ComparativeContext(BaseModel):
    """How this token compares to its sector/peers."""
    sector: str = Field(default="unknown")
    rank_in_sector: int = Field(default=0, description="1 = best in sector")
    sector_size: int = Field(default=0, description="Total tokens in sector")
    percentile: float = Field(default=0.5, ge=0, le=1)
    leader_score: float = Field(default=0, ge=0, le=1, description="Score of sector leader")


class MemoryContext(BaseModel):
    """Lessons from past similar trades."""
    similar_trades: int = Field(default=0, description="Past trades on similar tokens/narratives")
    win_rate: float = Field(default=0.5, ge=0, le=1)
    avg_return_pct: float = Field(default=0)
    best_outcome_pct: float = Field(default=0)
    worst_outcome_pct: float = Field(default=0)
    key_lesson: str = Field(default="")


# ── Enriched decision ─────────────────────────────────────────


class EnrichedDecision(CompactDecision):
    """Full trading decision with comparative context, momentum decomposition,
    position sizing rationale, and execution quality prediction.

    Designed to feed directly into AST's SafetyBreaker for final gatekeeping.
    """

    # Enriched dimensions
    narrative_lifecycle: NarrativeLifecycle = Field(default_factory=NarrativeLifecycle)
    momentum: MomentumDecomposition = Field(default_factory=MomentumDecomposition)
    sizing: PositionSizing = Field(default_factory=PositionSizing)
    execution_quality: ExecutionQuality = Field(default_factory=ExecutionQuality)
    comparative: ComparativeContext = Field(default_factory=ComparativeContext)
    memory: MemoryContext = Field(default_factory=MemoryContext)

    # Bridge metadata
    bridge_ready: bool = Field(default=False, description="True if this decision passed all gates and can be submitted to AST")
    bridge_rejection_reason: str = Field(default="", description="Why this was rejected for bridge submission")

    @property
    def is_bridge_ready(self) -> bool:
        """Full gate check for AST bridge submission.

        Must pass: should_trade + liquidity OK + conviction >= MEDIUM + no critical risks.
        """
        if not self.should_trade:
            return False
        if self.risk.overall == RiskLevel.CRITICAL:
            return False
        if self.risk.honeypot > 0.5:
            return False
        if self.execution.position_size_usd <= 0:
            return False
        if self.execution_quality.estimated_fill_pct < 0.5:
            return False
        return True

    def evaluate_and_mark(self) -> EnrichedDecision:
        """Run all gate checks and mark bridge_ready.

        Returns self for chaining.
        """
        if not self.should_trade:
            self.bridge_ready = False
            self.bridge_rejection_reason = f"should_trade=False (action={self.action.value}, conviction={self.conviction.value})"
            return self

        if self.risk.overall == RiskLevel.CRITICAL:
            self.bridge_ready = False
            self.bridge_rejection_reason = "CRITICAL risk level"
            return self

        if self.risk.honeypot > 0.5:
            self.bridge_ready = False
            self.bridge_rejection_reason = f"honeypot risk={self.risk.honeypot:.0%}"
            return self

        if self.execution.position_size_usd <= 0:
            self.bridge_ready = False
            self.bridge_rejection_reason = "position_size_usd <= 0"
            return self

        if self.execution_quality.estimated_fill_pct < 0.5:
            self.bridge_ready = False
            self.bridge_rejection_reason = f"estimated_fill_pct={self.execution_quality.estimated_fill_pct:.0%} < 50%"
            return self

        self.bridge_ready = True
        self.bridge_rejection_reason = ""
        return self

    @property
    def trading_card(self) -> str:
        """Rich one-line trading card for LLM context injection."""
        onchain_emoji = "🟢" if self.onchain_score.score > 0.7 else "🟡" if self.onchain_score.score > 0.4 else "🔴"
        sent_emoji = "🟢" if self.sentiment_score.score > 0.6 else "🟡" if self.sentiment_score.score > 0.4 else "🔴"
        narr_emoji = "🟢" if self.narrative_score.score > 0.6 else "🟡" if self.narrative_score.score > 0.4 else "🔴"
        tech_emoji = "🟢" if self.technical_score.score > 0.6 else "🟡" if self.technical_score.score > 0.4 else "🔴"

        parts = [
            f"{self.token[:12]}@{self.chain}",
            f"{self.action.value.upper()}",
            f"conv:{self.conviction.value}",
            f"conf:{self.confidence:.0%}",
            f"{onchain_emoji}onchain:{self.onchain_score.score:.0%}",
            f"{sent_emoji}sent:{self.sentiment_score.score:.0%}",
            f"{narr_emoji}narr:{self.narrative_score.score:.0%}",
            f"{tech_emoji}tech:{self.technical_score.score:.0%}",
            f"risk:{self.risk.overall.value}",
            f"${self.execution.position_size_usd:.0f}@{self.execution.entry_price_usd or 'MKT'}",
            f"tp:{self.execution.take_profit_pct:.0%}",
            f"sl:{self.execution.stop_loss_pct:.0%}",
        ]
        if self.momentum.convergence != "neutral":
            parts.append(f"mom:{self.momentum.convergence}")
        if self.narrative_lifecycle.phase not in ("unknown",):
            parts.append(f"narr_phase:{self.narrative_lifecycle.phase}")
        return " | ".join(parts)

    def to_bridge_signal(self) -> dict:
        """Convert to the JSON format expected by AST's /bridge/signal endpoint."""
        from cryptoagent.bridge.signal_builder import AgenticTradingSignal, Direction, Conviction as BridgeConviction, Rating

        direction_map = {
            Action.BUY: Direction.LONG,
            Action.SELL: Direction.CLOSE,
            Action.HOLD: Direction.CLOSE,
        }
        rating_map = {
            "buy": Rating.BUY,
            "overweight": Rating.OVERWEIGHT,
            "hold": Rating.HOLD,
            "underweight": Rating.UNDERWEIGHT,
            "sell": Rating.SELL,
        }
        conviction_map = {
            Conviction.LOW: BridgeConviction.LOW,
            Conviction.MEDIUM: BridgeConviction.MEDIUM,
            Conviction.HIGH: BridgeConviction.HIGH,
            Conviction.CONVICTION: BridgeConviction.CONVICTION,
        }

        signal = AgenticTradingSignal(
            token_address=self.token,
            chain=self.chain,
            direction=direction_map.get(self.action, Direction.CLOSE),
            rating=Rating.HOLD,  # Will be overridden below
            conviction=conviction_map.get(self.conviction, BridgeConviction.MEDIUM),
            rationale=self.thesis,
            time_horizon=self.execution.time_horizon.value,
            narrative_tags=self.narrative_score.flags if self.narrative_score.flags else [],
            entry_type=self.execution.entry_type,
            entry_price_usd=self.execution.entry_price_usd,
            position_size_usd=self.execution.position_size_usd,
            max_slippage_bps=self.execution.max_slippage_bps,
            stop_loss_pct=self.execution.stop_loss_pct,
            take_profit_pct=self.execution.take_profit_pct,
            mev_protection=self.execution_quality.mev_risk > 0.3,
            gas_strategy="fast" if self.execution_quality.mev_risk > 0.5 else "normal",
            risk_flags=[f"liquidity:{self.risk.liquidity:.0%}" if self.risk.liquidity > 0.3 else "",
                       f"mev:{self.risk.mev:.0%}" if self.risk.mev > 0.3 else "",
                       f"concentration:{self.risk.concentration:.0%}" if self.risk.concentration > 0.3 else ""],
        )
        return signal.to_dict() if hasattr(signal, 'to_dict') else signal.__dict__


# ── Builder from compact decision ─────────────────────────────


def enrich_decision(compact: CompactDecision, state: dict | None = None,
                    memory: MemoryContext | None = None) -> EnrichedDecision:
    """Enrich a CompactDecision with comparative context, momentum,
    position sizing, and execution quality.

    Args:
        compact: The base compact decision from analysis
        state: Optional full analysis state for additional extraction
        memory: Optional memory context from past trades
    """
    # ── Narrative lifecycle ──
    narrative_raw = (state or {}).get("narrative_report", "")
    lifecycle = NarrativeLifecycle()
    nl = narrative_raw.lower()
    if "growing" in nl or "building" in nl or "accelerating" in nl:
        lifecycle.phase = "accelerating"
        lifecycle.strength = 0.7
    elif "peak" in nl or "saturated" in nl:
        lifecycle.phase = "peak"
        lifecycle.strength = 0.9
    elif "exhausted" in nl or "declining" in nl or "fading" in nl:
        lifecycle.phase = "declining"
        lifecycle.strength = 0.3
    elif "emerging" in nl or "early" in nl:
        lifecycle.phase = "emerging"
        lifecycle.strength = 0.5
    if "inflow" in nl:
        lifecycle.rotation = "inflow"
    elif "outflow" in nl:
        lifecycle.rotation = "outflow"

    # ── Momentum decomposition ──
    technical_raw = (state or {}).get("technical_report", "")
    momentum = MomentumDecomposition()
    tl = technical_raw.lower()
    import re
    # Extract multi-timeframe changes
    for tf, field in [("1h", "scalp_1h"), ("6h", "intraday_6h"), ("24h", "swing_24h"), ("7d", "trend_7d")]:
        m = re.search(rf'{tf}.*?([+-]?[\d.]+)%', tl)
        if m:
            pct = float(m.group(1)) / 100
            setattr(momentum, field, max(-1, min(1, pct)))

    # Determine convergence
    signs = [1 if getattr(momentum, f) > 0.01 else -1 if getattr(momentum, f) < -0.01 else 0
             for f in ["scalp_1h", "intraday_6h", "swing_24h", "trend_7d"]]
    if all(s >= 0 for s in signs) and any(s > 0 for s in signs):
        momentum.convergence = "aligning_up"
    elif all(s <= 0 for s in signs) and any(s < 0 for s in signs):
        momentum.convergence = "aligning_down"
    elif signs[0] > 0 and any(s < 0 for s in signs[1:]):
        momentum.convergence = "diverging"  # short-term up, longer-term down
    elif signs[0] < 0 and any(s > 0 for s in signs[1:]):
        momentum.convergence = "diverging"

    if "rising" in tl or "accelerating" in tl:
        momentum.volume_trend = "rising"
    elif "falling" in tl or "declining" in tl:
        momentum.volume_trend = "falling"

    # ── Position sizing (Kelly-inspired) ──
    sizing = PositionSizing()
    if compact.confidence > 0 and compact.execution.take_profit_pct > 0:
        # Kelly: f* = p - (1-p)/(W/L) where p=confidence, W=take_profit, L=stop_loss
        p = compact.confidence
        W = compact.execution.take_profit_pct
        L = compact.execution.stop_loss_pct
        if L > 0:
            kelly = max(0, p - (1 - p) / (W / L))
            sizing.kelly_fraction = min(0.25, kelly)  # Half-Kelly cap at 25%
        else:
            sizing.kelly_fraction = 0

    # Liquidity constraint: max 2% of pool
    liquidity_usd = compact.execution.position_size_usd * 10  # rough estimate
    if liquidity_usd > 0:
        sizing.max_size_usd = liquidity_usd * 0.02  # 2% of pool
    else:
        sizing.max_size_usd = compact.execution.position_size_usd

    # Adjusted size: min(Kelly size, liquidity cap)
    kelly_size = compact.execution.position_size_usd * sizing.kelly_fraction if sizing.kelly_fraction > 0 else compact.execution.position_size_usd
    sizing.adjusted_size_usd = min(kelly_size, sizing.max_size_usd) if sizing.max_size_usd > 0 else kelly_size

    if sizing.kelly_fraction > 0:
        sizing.rationale = f"Kelly={sizing.kelly_fraction:.0%}, capped at {sizing.max_size_usd:.0f} (2% pool)"
    else:
        sizing.rationale = "Kelly fraction zero — edge insufficient for sized bet"

    # ── Execution quality ──
    eq = ExecutionQuality()
    # Estimate fill probability based on entry type and liquidity
    if compact.execution.entry_type == "market":
        eq.estimated_fill_pct = 0.99
        eq.estimated_slippage_bps = compact.execution.max_slippage_bps // 2
    elif compact.execution.entry_type == "limit":
        eq.estimated_fill_pct = 0.6  # Limit orders may not fill
        eq.estimated_slippage_bps = compact.execution.max_slippage_bps // 4
    else:
        eq.estimated_fill_pct = 0.85
        eq.estimated_slippage_bps = compact.execution.max_slippage_bps // 3

    # MEV risk based on chain and size
    if compact.chain in ("ethereum", "bsc"):
        eq.mev_risk = min(0.8, compact.execution.position_size_usd / 100000)
    elif compact.chain == "solana":
        eq.mev_risk = min(0.4, compact.execution.position_size_usd / 200000)
    else:
        eq.mev_risk = 0.2

    eq.recommended_mempool = "flashbots" if eq.mev_risk > 0.4 else "private" if eq.mev_risk > 0.2 else "public"
    eq.gas_estimate_usd = {"ethereum": 15.0, "bsc": 0.50, "arbitrum": 1.0, "base": 0.30, "solana": 0.01}.get(compact.chain, 2.0)

    # ── Comparative context ──
    comp = ComparativeContext()
    onchain_flags = compact.onchain_score.flags
    if "ai" in str(onchain_flags).lower() or "ai" in (state or {}).get("narrative_report", "").lower():
        comp.sector = "ai"
    elif "meme" in str(onchain_flags).lower():
        comp.sector = "memecoin"
    elif "defi" in str(onchain_flags).lower():
        comp.sector = "defi"
    else:
        comp.sector = "general"

    # ── Memory ──
    mem = memory or MemoryContext()

    # ── Build enriched ──
    enriched = EnrichedDecision(
        **compact.model_dump(),
        narrative_lifecycle=lifecycle,
        momentum=momentum,
        sizing=sizing,
        execution_quality=eq,
        comparative=comp,
        memory=mem,
    )

    # Run gate checks
    enriched.evaluate_and_mark()

    return enriched
