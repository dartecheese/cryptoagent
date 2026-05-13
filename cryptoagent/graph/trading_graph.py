"""TradingGraph — main orchestrator for CryptoAgent.

Crypto-adapted from TradingAgents' trading_graph.py.

Pulls together:
- LLM clients (quick + deep thinking)
- Tool definitions (from agent_utils)
- LangGraph setup (from setup.py)
- State propagation (from propagation.py)
- Trading memory (from memory.py)

Usage:
    from cryptoagent.graph.trading_graph import CryptoAgentGraph
    from cryptoagent.config import CRYPTOAGENT_DEFAULT_CONFIG

    cg = CryptoAgentGraph(config=CRYPTOAGENT_DEFAULT_CONFIG.copy())
    final_state, decision = cg.propagate("0xTOKEN", "ethereum")
    print(decision)
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

from cryptoagent.agents.utils.agent_utils import (
    get_token_metadata,
    get_token_price_ohlcv,
    get_onchain_metrics,
    get_crypto_news,
    get_narrative_data,
    get_social_sentiment,
)
from cryptoagent.agents.utils.agent_states import AgentState
from cryptoagent.config import CRYPTOAGENT_DEFAULT_CONFIG
from cryptoagent.dataflows.config import set_config as set_data_config

from cryptoagent.graph.conditional_logic import ConditionalLogic
from cryptoagent.graph.setup import GraphSetup
from cryptoagent.graph.propagation import Propagator
from cryptoagent.graph.memory import TradingMemoryLog


class CryptoAgentGraph:
    """Main orchestrator for the crypto trading agents framework.

    Mirrors TradingAgentsGraph from TradingAgents but adapted for crypto:
    - Token address + chain instead of ticker + date
    - Crypto-specific analyst team
    - Crypto risk dimensions (MEV, liquidity, rug)
    - AST execution bridge
    """

    def __init__(
        self,
        selected_analysts: list[str] | None = None,
        debug: bool = False,
        config: Dict[str, Any] | None = None,
        callbacks: Optional[List] = None,
    ):
        self.debug = debug
        self.config = config or CRYPTOAGENT_DEFAULT_CONFIG.copy()
        self.callbacks = callbacks or []

        if selected_analysts is None:
            selected_analysts = ["onchain", "sentiment", "narrative", "technical"]

        # Update data layer config
        set_data_config(self.config)

        # Create directories
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        # ── LLM Clients ─────────────────────────────────
        from langchain_openai import ChatOpenAI

        llm_kwargs = {}
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        provider = self.config.get("llm_provider", "openai")
        deep_model = self.config["deep_think_llm"]
        quick_model = self.config["quick_think_llm"]
        backend_url = self.config.get("backend_url")
        api_key = self.config.get("api_key")

        deep_kwargs = {"model": deep_model, **llm_kwargs}
        quick_kwargs = {"model": quick_model, **llm_kwargs}

        # Provider-specific base URLs
        provider_urls = {
            "deepseek": "https://api.deepseek.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "openai": None,
        }
        if backend_url:
            deep_kwargs["base_url"] = backend_url
            quick_kwargs["base_url"] = backend_url
        elif provider in provider_urls and provider_urls[provider]:
            deep_kwargs["base_url"] = provider_urls[provider]
            quick_kwargs["base_url"] = provider_urls[provider]

        # API key from config or env
        if api_key:
            deep_kwargs["api_key"] = api_key
            quick_kwargs["api_key"] = api_key

        self.deep_thinking_llm = ChatOpenAI(**deep_kwargs)
        self.quick_thinking_llm = ChatOpenAI(**quick_kwargs)

        # ── Trading Memory ──────────────────────────────
        self.memory_log = TradingMemoryLog(self.config)

        # ── Tool Nodes ──────────────────────────────────
        from langgraph.prebuilt import ToolNode

        self.tool_nodes = {
            "onchain": ToolNode([get_token_metadata, get_onchain_metrics]),
            "sentiment": ToolNode([get_social_sentiment, get_crypto_news]),
            "narrative": ToolNode([get_narrative_data, get_token_metadata]),
            "technical": ToolNode([get_token_price_ohlcv, get_token_metadata]),
        }

        # ── Graph Components ────────────────────────────
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.conditional_logic,
            provider=provider,
        )
        self.propagator = Propagator(
            max_recur_limit=self.config.get("max_recur_limit", 100),
        )

        # ── Build Graph ─────────────────────────────────
        self.workflow = self.graph_setup.setup_graph(selected_analysts)
        self.graph = self.workflow.compile()
        self.selected_analysts = selected_analysts

        # State
        self.curr_state = None
        self._last_compact = None
        self.token_of_interest = None
        self.chain = None

    def propagate(
        self,
        token: str,
        chain: str = "ethereum",
    ) -> tuple[Dict, str]:
        """Run the full trading agent analysis for a token.

        Args:
            token: Token contract address or symbol
            chain: Blockchain context

        Returns:
            Tuple of (final_state, final_decision_text)
        """
        self.token_of_interest = token
        self.chain = chain

        # Load past context from trading memory
        past_context = self.memory_log.get_recent_context(token=token)

        # Create initial state
        initial_state = self.propagator.create_initial_state(
            token=token,
            chain=chain,
            past_context=past_context,
        )

        # Run the graph
        graph_args = self.propagator.get_graph_args(self.callbacks)

        if self.debug:
            logger.info(f"Starting analysis for {token} on {chain}")
            logger.info(f"Config: provider={self.config['llm_provider']}, "
                        f"debate_rounds={self.config['max_debate_rounds']}, "
                        f"risk_rounds={self.config['max_risk_discuss_rounds']}")

        final_state = None
        for state in self.graph.stream(initial_state, **graph_args):
            final_state = state
            if self.debug:
                node_name = list(state.keys())[0] if state else "unknown"
                logger.info(f"Completed node: {node_name}")

        self.curr_state = final_state

        # Extract the final decision from the state dict
        if final_state:
            decision = final_state.get("final_trade_decision", "")
            self._last_compact = None  # Reset cache
            return final_state, decision

        return {}, "No decision produced."

    def get_compact_decision(self) -> Optional["CompactDecision"]:
        """Build a CompactDecision from the current analysis state.

        Returns a machine-actionable ~500 byte decision suitable for
        automated agentic trading. Returns None if no analysis has been run.
        """
        if self._last_compact is not None:
            return self._last_compact

        if not self.curr_state:
            return None

        from cryptoagent.agents.compact_decision import build_compact_decision
        self._last_compact = build_compact_decision(self.curr_state)
        return self._last_compact

    def get_report(self) -> str:
        """Build a comprehensive markdown report from the current state."""
        if not self.curr_state:
            return "No analysis has been run yet."

        state = self.curr_state if isinstance(self.curr_state, dict) else {}

        sections = [
            f"# CryptoAgent Analysis Report",
            f"**Token**: `{state.get('token_of_interest', 'N/A')}`",
            f"**Chain**: {state.get('chain', 'N/A')}",
            f"**Analyzed**: {state.get('analysis_timestamp', 'N/A')}",
            "",
            "---",
            "",
            state.get("onchain_report", "*No on-chain analysis*"),
            "",
            state.get("sentiment_report", "*No sentiment analysis*"),
            "",
            state.get("narrative_report", "*No narrative analysis*"),
            "",
            state.get("technical_report", "*No technical analysis*"),
            "",
            "---",
            "",
            "## Research Manager Decision",
            state.get("investment_plan", "*No investment plan*"),
            "",
            "## Trader Execution Plan",
            state.get("trader_investment_plan", "*No trader plan*"),
            "",
            "## Portfolio Manager Final Decision",
            state.get("final_trade_decision", "*No final decision*"),
        ]
        return "\n\n".join(sections)

    def save_report(self, filepath: str | None = None) -> Path:
        """Save the analysis report to disk."""
        if filepath is None:
            timestamp = self.curr_state.get("analysis_timestamp", "") if self.curr_state else ""
            safe_ts = timestamp.replace(":", "-") if timestamp else "unknown"
            token_slug = str(self.token_of_interest)[:12] if self.token_of_interest else "unknown"
            filepath = str(Path(self.config["results_dir"]) / f"{token_slug}_{safe_ts}.md")

        report = self.get_report()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report)
        return path
