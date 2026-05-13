"""Technical Analyst — price action, trends, volume, liquidity depth."""

from cryptoagent.agents.analysts.analyst_base import make_analyst_node
from cryptoagent.agents.utils.agent_utils import get_token_price_ohlcv, get_token_metadata


def create_technical_analyst(llm):
    tools = [get_token_price_ohlcv, get_token_metadata]

    system_message = (
        "You are a Technical Analyst. Fetch data then output ONLY valid JSON:\n"
        '{{"score": 0.0-1.0 (bullish=1, bearish=0), "confidence": 0.0-1.0, '
        '"flags": ["breakout"|"breakdown"|"overbought"|"oversold"|"dead_cat"|"none"], '
        '"trend": "up"|"down"|"flat", '
        '"change_1h_pct": number, "change_6h_pct": number, "change_24h_pct": number, '
        '"volume_trend": "rising"|"steady"|"falling", '
        '"summary": "1-line finding"}}'
    )
    return make_analyst_node(llm, tools, system_message, "technical_report")
