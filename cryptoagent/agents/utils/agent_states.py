"""LangGraph agent state types — crypto-adapted from TradingAgents."""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import MessagesState


class InvestDebateState(TypedDict):
    """State for the Bull vs Bear researcher debate."""
    bull_history: Annotated[str, "Bullish conversation history"]
    bear_history: Annotated[str, "Bearish conversation history"]
    history: Annotated[str, "Full debate history"]
    current_response: Annotated[str, "Latest response"]
    judge_decision: Annotated[str, "Research Manager's verdict"]
    count: Annotated[int, "Number of debate rounds completed"]


class RiskDebateState(TypedDict):
    """State for the Aggressive/Neutral/Conservative risk debate."""
    aggressive_history: Annotated[str, "Aggressive analyst history"]
    conservative_history: Annotated[str, "Conservative analyst history"]
    neutral_history: Annotated[str, "Neutral analyst history"]
    history: Annotated[str, "Full risk debate history"]
    latest_speaker: Annotated[str, "Which analyst spoke last"]
    current_aggressive_response: Annotated[str, "Latest aggressive response"]
    current_conservative_response: Annotated[str, "Latest conservative response"]
    current_neutral_response: Annotated[str, "Latest neutral response"]
    judge_decision: Annotated[str, "Portfolio Manager's verdict"]
    count: Annotated[int, "Number of risk debate rounds completed"]


class AgentState(MessagesState):
    """Top-level state that flows through the LangGraph pipeline.

    Crypto-specific: 'token_of_interest' replaces 'company_of_interest'.
    'chain' is added so all agents know which blockchain context.
    """

    token_of_interest: Annotated[
        str, "Token contract address or symbol being analyzed"
    ]
    chain: Annotated[
        str, "Blockchain context: ethereum, base, arbitrum, solana, bsc"
    ]
    analysis_timestamp: Annotated[
        str, "ISO timestamp of when this analysis was requested"
    ]
    sender: Annotated[str, "Agent that sent the current message"]

    # ── Analyst Reports ──
    onchain_report: Annotated[str, "Report from the On-Chain Analyst"]
    sentiment_report: Annotated[str, "Report from the Sentiment Analyst"]
    narrative_report: Annotated[str, "Report from the Narrative Analyst"]
    technical_report: Annotated[str, "Report from the Technical Analyst"]

    # ── Researcher Team ──
    investment_debate_state: Annotated[
        InvestDebateState, "State tracking for the bull/bear debate"
    ]
    investment_plan: Annotated[str, "Research Manager's investment plan"]

    # ── Trader ──
    trader_investment_plan: Annotated[str, "Trader's concrete execution plan"]

    # ── Risk Management ──
    risk_debate_state: Annotated[
        RiskDebateState, "State tracking for the risk debate"
    ]
    final_trade_decision: Annotated[str, "Portfolio Manager's final decision"]

    # ── Memory ──
    past_context: Annotated[
        str, "Past trading memory injected at run start for learning"
    ]
