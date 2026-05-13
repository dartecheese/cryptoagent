"""Technical Analyst — charts and price action analysis.

Most portable from TradingAgents' Market Analyst. Adapted for crypto:
- Uses DexScreener OHLCV data instead of Yahoo Finance
- Same TA tools: RSI, MACD, moving averages, volume analysis
- Adds crypto-specific: DEX pair analysis, liquidity depth
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from cryptoagent.agents.utils.agent_utils import (
    build_token_context,
    get_language_instruction,
    get_token_price_ohlcv,
    get_token_metadata,
)


def create_technical_analyst(llm):
    """Create the technical analyst node."""

    def technical_analyst_node(state):
        token = state["token_of_interest"]
        chain = state["chain"]
        token_context = build_token_context(token, chain)

        tools = [
            get_token_price_ohlcv,
            get_token_metadata,
        ]

        system_message = (
            "You are a crypto Technical Analyst. Your job is to analyze price "
            "action and identify trade setups using technical analysis.\n\n"
            "Analyze these dimensions:\n"
            "1. **Trend Analysis**: What is the overall trend? Use multiple timeframes "
            "(1h, 4h, 1d). Is the trend strong or weakening?\n"
            "2. **Support/Resistance**: Identify key levels. Where is price relative "
            "to recent highs/lows?\n"
            "3. **Momentum**: Look at price changes across timeframes (1h, 6h, 24h). "
            "Is momentum accelerating or decelerating?\n"
            "4. **Volume Analysis**: Is volume confirming the price move? Rising volume "
            "on up moves = bullish. Falling volume on up moves = warning.\n"
            "5. **Liquidity Depth**: Check DEX liquidity. Thin liquidity = high impact "
            "cost and slippage risk.\n"
            "6. **Buy/Sell Pressure**: Compare buy vs sell transaction counts and volumes.\n\n"
            "Provide a clear TECHNICAL RATING: 🟢 Bullish Setup / 🟡 Neutral / 🔴 Bearish Setup. "
            "Include specific entry/exit levels if you see them. End with a Markdown table."
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
            "technical_report": report,
        }

    return technical_analyst_node
