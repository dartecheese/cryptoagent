"""Propagation — state initialization and graph execution.

Crypto-adapted from TradingAgents' propagation.py.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from cryptoagent.agents.utils.agent_states import InvestDebateState, RiskDebateState


class Propagator:
    """Handles state initialization and graph execution."""

    def __init__(self, max_recur_limit: int = 100):
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self,
        token: str,
        chain: str,
        past_context: str = "",
    ) -> Dict[str, Any]:
        """Create the initial state for the agent graph.

        Args:
            token: Token contract address or symbol
            chain: Blockchain (ethereum, base, arbitrum, solana, bsc)
            past_context: Previous trading memory for this token/similar tokens
        """
        return {
            "messages": [("human", token)],
            "token_of_interest": token,
            "chain": chain,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "past_context": past_context,
            "investment_debate_state": InvestDebateState({
                "bull_history": "",
                "bear_history": "",
                "history": "",
                "current_response": "",
                "judge_decision": "",
                "count": 0,
            }),
            "risk_debate_state": RiskDebateState({
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "history": "",
                "latest_speaker": "",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "judge_decision": "",
                "count": 0,
            }),
            "onchain_report": "",
            "sentiment_report": "",
            "narrative_report": "",
            "technical_report": "",
        }

    def get_graph_args(self, callbacks: Optional[List] = None) -> Dict[str, Any]:
        """Get arguments for the compiled graph's invoke/stream call."""
        config = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "values",
            "config": config,
        }
