"""Agent exports — mirror TradingAgents' __init__.py pattern."""

from cryptoagent.agents.utils.agent_utils import create_msg_delete
from cryptoagent.agents.utils.agent_states import AgentState, InvestDebateState, RiskDebateState

from cryptoagent.agents.analysts.onchain_analyst import create_onchain_analyst
from cryptoagent.agents.analysts.sentiment_analyst import create_sentiment_analyst
from cryptoagent.agents.analysts.narrative_analyst import create_narrative_analyst
from cryptoagent.agents.analysts.technical_analyst import create_technical_analyst

from cryptoagent.agents.researchers.bull_researcher import create_bull_researcher
from cryptoagent.agents.researchers.bear_researcher import create_bear_researcher

from cryptoagent.agents.managers.research_manager import create_research_manager
from cryptoagent.agents.managers.portfolio_manager import create_portfolio_manager

from cryptoagent.agents.trader.trader import create_trader

from cryptoagent.agents.risk.aggressive import create_aggressive_debator
from cryptoagent.agents.risk.neutral import create_neutral_debator
from cryptoagent.agents.risk.conservative import create_conservative_debator

__all__ = [
    "AgentState",
    "InvestDebateState",
    "RiskDebateState",
    "create_msg_delete",
    "create_onchain_analyst",
    "create_sentiment_analyst",
    "create_narrative_analyst",
    "create_technical_analyst",
    "create_bull_researcher",
    "create_bear_researcher",
    "create_research_manager",
    "create_trader",
    "create_aggressive_debator",
    "create_neutral_debator",
    "create_conservative_debator",
    "create_portfolio_manager",
]
