"""On-Chain Analyst — contract security, holders, liquidity, tax, age."""

from cryptoagent.agents.analysts.analyst_base import make_analyst_node
from cryptoagent.agents.utils.agent_utils import get_token_metadata, get_onchain_metrics


def create_onchain_analyst(llm):
    tools = [get_token_metadata, get_onchain_metrics]

    system_message = (
        "You are an On-Chain Analyst. Fetch data then output ONLY valid JSON:\n"
        '{{"score": 0.0-1.0, "confidence": 0.0-1.0, '
        '"flags": ["honeypot"|"proxy"|"low_liq"|"concentrated"|"new_token"|"tax"|"none"], '
        '"liquidity_usd": number, "holders": number, "age_days": number, '
        '"summary": "1-line finding"}}'
    )
    return make_analyst_node(llm, tools, system_message, "onchain_report")
