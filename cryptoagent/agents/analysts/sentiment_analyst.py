"""Sentiment Analyst — buy/sell ratio, volume, price action, catalysts."""

from cryptoagent.agents.analysts.analyst_base import make_analyst_node
from cryptoagent.agents.utils.agent_utils import get_social_sentiment, get_crypto_news


def create_sentiment_analyst(llm):
    tools = [get_social_sentiment, get_crypto_news]

    system_message = (
        "You are a Sentiment Analyst. Fetch data then output ONLY valid JSON:\n"
        '{{"score": 0.0-1.0 (bullish=1, bearish=0), "confidence": 0.0-1.0, '
        '"flags": ["sell_pressure"|"buy_surge"|"low_volume"|"catalyst"|"none"], '
        '"buy_sell_ratio": number, "volume_24h_usd": number, "price_change_24h_pct": number, '
        '"summary": "1-line finding"}}'
    )
    return make_analyst_node(llm, tools, system_message, "sentiment_report")
