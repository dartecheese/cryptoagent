"""Bridge to AST (Asymmetric Strike Team) Rust execution layer.

Converts CryptoAgent's LLM-generated Portfolio Decision into a structured
AgenticTradingSignal that the Rust AST process can consume via IPC or HTTP.

The bridge is deliberately one-way: LLM research → AST execution.
The Rust layer is the final authority — it runs SafetyBreaker checks
before any trade hits the chain.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ── Signal Types (mirrors AST's crates/ast-core/src/types.rs) ───────


class Direction(str, Enum):
    LONG = "Long"
    SHORT = "Short"
    CLOSE = "Close"


class Conviction(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CONVICTION = "Conviction"


class Rating(str, Enum):
    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class EntryType(str, Enum):
    MARKET = "Market"
    LIMIT = "Limit"
    TWAP = "TWAP"


class GasStrategy(str, Enum):
    FAST = "Fast"
    NORMAL = "Normal"
    SLOW = "Slow"


@dataclass
class AgenticTradingSignal:
    """The canonical signal passed from LLM Research Layer to AST Execution Layer."""

    token_address: str
    chain: str
    direction: Direction
    rating: Rating
    conviction: Conviction

    # Thesis
    rationale: str
    time_horizon: str  # "Scalp", "Intraday", "Swing", "Position", "LongTerm"
    narrative_tags: list[str] = field(default_factory=list)

    # Execution parameters
    entry_type: Optional[str] = None
    entry_price_usd: Optional[float] = None
    position_size_usd: Optional[float] = None
    max_slippage_bps: Optional[int] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    mev_protection: bool = False
    gas_strategy: str = "Normal"

    # Risk flags for AST SafetyBreaker to verify
    risk_flags: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize to JSON for IPC/HTTP bridge."""
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_portfolio_decision(
        cls,
        decision: "PortfolioDecision",
        trader_proposal: "TraderProposal",
        token: str,
        chain: str,
    ) -> "AgenticTradingSignal":
        """Build an AgenticTradingSignal from CryptoAgent structured outputs.

        Args:
            decision: PortfolioDecision from the Portfolio Manager
            trader_proposal: TraderProposal from the Crypto Trader
            token: Token contract address
            chain: Blockchain context
        """
        from cryptoagent.agents.schemas import PortfolioDecision, TraderProposal

        # Map PortfolioRating + TraderAction to Direction
        direction_map = {
            ("Buy", "Buy"): Direction.LONG,
            ("Overweight", "Buy"): Direction.LONG,
            ("Hold", "Hold"): Direction.CLOSE,
            ("Underweight", "Sell"): Direction.CLOSE,
            ("Sell", "Sell"): Direction.CLOSE,
        }
        direction = direction_map.get(
            (decision.rating.value, trader_proposal.action.value),
            Direction.CLOSE,
        )

        return cls(
            token_address=token,
            chain=chain,
            direction=direction,
            rating=Rating(decision.rating.value),
            conviction=Conviction.HIGH,  # default, refine later
            rationale=decision.investment_thesis,
            time_horizon=decision.time_horizon.value,
            narrative_tags=decision.narrative_tags,
            entry_type=trader_proposal.entry_type.value if trader_proposal.entry_type else None,
            entry_price_usd=trader_proposal.entry_price_usd,
            position_size_usd=trader_proposal.position_size_usd,
            max_slippage_bps=trader_proposal.max_slippage_bps,
            stop_loss_pct=trader_proposal.stop_loss_pct,
            take_profit_pct=trader_proposal.take_profit_pct,
            risk_flags=[rf.value for rf in decision.risk_flags],
        )


# ── AST Client ──────────────────────────────────────────────────────


class ASTClient:
    """Client for communicating with the AST Rust execution process.

    Supports two modes:
    - IPC (Unix domain socket): faster, same-machine
    - HTTP: simpler, works across machines

    The Rust process listens for TradingSignals, runs them through
    SafetyBreaker → Actuary → Slinger → Reaper, and returns execution results.
    """

    def __init__(self, config: dict):
        self.ipc_path = config.get("ast_ipc_path", "/tmp/ast-execution.sock")
        self.http_url = config.get("ast_http_url", "http://localhost:9090")
        self.enabled = config.get("ast_execution_enabled", False)

    async def submit_signal(self, signal: AgenticTradingSignal) -> dict:
        """Submit a trading signal to the AST Rust process.

        Returns the execution result from AST.
        """
        if not self.enabled:
            return {"status": "disabled", "reason": "AST execution not enabled"}

        import httpx

        payload = signal.to_json()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.http_url}/signal",
                    content=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30.0,
                )
                if resp.status_code == 200:
                    return resp.json()
                return {
                    "status": "error",
                    "code": resp.status_code,
                    "body": resp.text,
                }
        except Exception as e:
            logger.error(f"AST bridge error: {e}")
            return {"status": "connection_error", "error": str(e)}

    def submit_signal_sync(self, signal: AgenticTradingSignal) -> dict:
        """Synchronous wrapper for submit_signal."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.submit_signal(signal))
