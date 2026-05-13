"""Sentiment Analyst — gauges market mood from trading activity and social signals.

Crypto-adapted: uses buy/sell ratio from DexScreener as proxy for sentiment,
plus news mentions from CryptoPanic. Extensible to Twitter/Discord/Telegram.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from cryptoagent.agents.utils.agent_utils import (
    build_token_context,
    get_language_instruction,
    get_social_sentiment,
    get_crypto_news,
)


def create_sentiment_analyst(llm):
    """Create the sentiment analyst node."""

    def sentiment_analyst_node(state):
        token = state["token_of_interest"]
        chain = state["chain"]
        token_context = build_token_context(token, chain)

        tools = [
            get_social_sentiment,
            get_crypto_news,
        ]

        system_message = (
            "You are a crypto Sentiment Analyst. Gauge market mood concisely.\n\n"
            "REQUIRED OUTPUT FORMAT (terse):\n"
            "1. Buy/Sell Ratio: [number] → [bullish|bearish|neutral]\n"
            "2. Volume 24h: [USD] → [accelerating|steady|declining]\n"
            "3. Price Change 24h: [%]\n"
            "4. Key Catalysts: [list or 'none detected']\n\n"
            "END WITH EXACTLY:\n"
            "SENTIMENT: [BULLISH|NEUTRAL|BEARISH]\n"
            "CONFIDENCE: [HIGH|MEDIUM|LOW]"
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK; another assistant with different tools"
                " will help where you left off. Execute what you can to make progress."
                " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}"
                " {token_context}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))
        prompt = prompt.partial(token_context=token_context)

        chain_prompt = prompt | llm.bind_tools(tools)
        result = chain_prompt.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "sentiment_report": report,
        }

    return sentiment_analyst_node
