"""Pydantic schemas for structured agent outputs — crypto-adapted.

Mirrors TradingAgents' schemas but with crypto-specific fields:
- Token address + chain instead of ticker
- On-chain metrics instead of fundamentals
- DeFi execution parameters instead of equity order types
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Shared rating types ────────────────────────────────────────────


class PortfolioRating(str, Enum):
    """5-tier rating used by Research Manager and Portfolio Manager."""
    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction."""
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


class ConvictionLevel(str, Enum):
    """How confident is the thesis?"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CONVICTION = "Conviction"


class TimeHorizon(str, Enum):
    """Crypto-appropriate time horizons."""
    SCALP = "Scalp"          # < 1 hour
    INTRADAY = "Intraday"    # 1-24 hours
    SWING = "Swing"          # 1-7 days
    POSITION = "Position"    # 1-4 weeks
    LONG_TERM = "LongTerm"   # > 1 month


class EntryType(str, Enum):
    """How to enter the position."""
    MARKET = "Market"
    LIMIT = "Limit"
    TWAP = "TWAP"


class GasStrategy(str, Enum):
    """Transaction priority."""
    FAST = "Fast"
    NORMAL = "Normal"
    SLOW = "Slow"


class RiskFlag(str, Enum):
    """LLM-identified risks that AST's deterministic layer should verify."""
    HONEYPOT_SUSPECT = "HoneypotSuspect"
    RUG_POTENTIAL = "RugPotential"
    WASH_TRADING = "WashTrading"
    LOW_LIQUIDITY = "LowLiquidity"
    HIGH_CONCENTRATION = "HighConcentration"
    MEV_RISK = "MevRisk"
    NARRATIVE_EXHAUSTED = "NarrativeExhausted"
    DEV_ABANDONED = "DevAbandoned"
    CEX_LISTING_RISK = "CexListingRisk"


# ── Research Manager ───────────────────────────────────────────────


class ResearchPlan(BaseModel):
    """Structured investment plan from the Research Manager."""

    recommendation: PortfolioRating = Field(
        description="Investment recommendation: Buy/Overweight/Hold/Underweight/Sell."
    )
    rationale: str = Field(
        description="Why this recommendation — key arguments from the debate, which side won."
    )
    strategic_actions: str = Field(
        description="Concrete steps to execute, including entry strategy, sizing, and timing."
    )
    conviction: ConvictionLevel = Field(
        description="Confidence level in this recommendation."
    )


def render_research_plan(plan: ResearchPlan) -> str:
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        f"**Conviction**: {plan.conviction.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ── Trader ─────────────────────────────────────────────────────────


class TraderProposal(BaseModel):
    """Structured transaction proposal from the Crypto Trader."""

    action: TraderAction = Field(
        description="Transaction direction: Buy/Hold/Sell."
    )
    reasoning: str = Field(
        description="The case for this action, anchored in analyst reports and research plan."
    )
    chain: str = Field(
        description="Target blockchain: ethereum, base, arbitrum, solana, bsc."
    )
    entry_type: EntryType = Field(
        description="How to enter: Market/Limit/TWAP."
    )
    entry_price_usd: Optional[float] = Field(
        default=None, description="Target entry price in USD."
    )
    stop_loss_pct: Optional[float] = Field(
        default=None, description="Stop loss as percentage (e.g. 0.15 = 15%)."
    )
    take_profit_pct: Optional[float] = Field(
        default=None, description="Take profit as percentage (e.g. 1.0 = 100%)."
    )
    position_size_usd: Optional[float] = Field(
        default=None, description="Position size in USD."
    )
    max_slippage_bps: Optional[int] = Field(
        default=None, description="Max slippage in basis points (e.g. 300 = 3%)."
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    parts = [
        f"**Action**: {proposal.action.value}",
        f"**Chain**: {proposal.chain}",
        f"**Entry Type**: {proposal.entry_type.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price_usd is not None:
        parts.append(f"**Entry Price**: ${proposal.entry_price_usd:.6f}")
    if proposal.stop_loss_pct is not None:
        parts.append(f"**Stop Loss**: {proposal.stop_loss_pct * 100:.1f}%")
    if proposal.take_profit_pct is not None:
        parts.append(f"**Take Profit**: {proposal.take_profit_pct * 100:.1f}%")
    if proposal.position_size_usd is not None:
        parts.append(f"**Position Size**: ${proposal.position_size_usd:,.2f}")
    if proposal.max_slippage_bps is not None:
        parts.append(f"**Max Slippage**: {proposal.max_slippage_bps / 100:.1f}%")
    parts.extend(["", f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**"])
    return "\n".join(parts)


# ── Portfolio Manager ──────────────────────────────────────────────


class PortfolioDecision(BaseModel):
    """Final structured output from the Portfolio Manager."""

    rating: PortfolioRating = Field(
        description="Final position rating: Buy/Overweight/Hold/Underweight/Sell."
    )
    executive_summary: str = Field(
        description="Concise action plan: entry, sizing, risk levels, time horizon."
    )
    investment_thesis: str = Field(
        description="Detailed reasoning anchored in analyst evidence and risk debate."
    )
    price_target: Optional[float] = Field(
        default=None, description="Target price in USD."
    )
    time_horizon: TimeHorizon = Field(
        description="Recommended holding period."
    )
    risk_flags: list[RiskFlag] = Field(
        default_factory=list,
        description="Risk factors for AST's deterministic verification layer."
    )
    narrative_tags: list[str] = Field(
        default_factory=list,
        description="Relevant narratives (e.g. 'AI agents', 'memecoin season')."
    )


def render_pm_decision(decision: PortfolioDecision) -> str:
    parts = [
        f"**Rating**: {decision.rating.value}",
        f"**Time Horizon**: {decision.time_horizon.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.append(f"**Price Target**: ${decision.price_target:.6f}")
    if decision.narrative_tags:
        parts.append(f"**Narratives**: {', '.join(decision.narrative_tags)}")
    if decision.risk_flags:
        parts.append(f"**Risk Flags**: {', '.join(r.value for r in decision.risk_flags)}")
    return "\n".join(parts)
