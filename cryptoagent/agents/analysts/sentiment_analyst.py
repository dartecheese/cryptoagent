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
            "You are a crypto Sentiment Analyst. Your job is to gauge market "
            "sentiment around a token using available data.\n\n"
            "Analyze these dimensions:\n"
            "1. **Trading Activity Sentiment**: Look at buy/sell ratio from recent "
            "DEX activity. Heavy buying > 2:1 ratio is bullish. Heavy selling < 0.5:1 "
            "is bearish.\n"
            "2. **Volume Acceleration**: Is 24h volume growing or shrinking? Compare "
            "current volume to recent averages.\n"
            "3. **News Sentiment**: What are recent headlines saying? Any catalysts "
            "(listings, partnerships, upgrades) or risks (hacks, exploits, FUD)?\n"
            "4. **Social Signals**: Is there growing interest? Check trending data.\n"
            "5. **Narrative Alignment**: Is this token riding a hot narrative or being ignored?\n\n"
            "Provide a clear SENTIMENT RATING: 🟢 Bullish / 🟡 Neutral / 🔴 Bearish "
            "with specific evidence. End with a Markdown table."
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
