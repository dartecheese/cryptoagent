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
            "You are a crypto Technical Analyst. Analyze price action concisely.\n\n"
            "REQUIRED OUTPUT FORMAT (terse):\n"
            "1. Trend (1h/6h/24h): [UP/DOWN/FLAT] with % change\n"
            "2. Volume: [USD 24h] — [rising/falling/steady]\n"
            "3. Key Level: nearest support/resistance\n"
            "4. Pattern: [accumulation/distribution/breakout/breakdown/none]\n"
            "5. Liquidity Depth: [USD] — slippage risk?\n\n"
            "END WITH EXACTLY:\n"
            "SETUP: [BULLISH|NEUTRAL|BEARISH]\n"
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
            "technical_report": report,
        }

    return technical_analyst_node
