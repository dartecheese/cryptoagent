"""On-Chain Analyst — evaluates token fundamentals via on-chain data.

Replaces TradingAgents' Fundamentals Analyst. Analyzes:
- Token contract security (honeypot, proxy, mint authority)
- Holder distribution and concentration
- Liquidity depth and lock status
- Protocol metrics (TVL, revenue) from DeFiLlama
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from cryptoagent.agents.utils.agent_utils import (
    build_token_context,
    get_language_instruction,
    get_token_metadata,
    get_onchain_metrics,
)


def create_onchain_analyst(llm):
    """Create the on-chain analyst node for the trading graph."""

    def onchain_analyst_node(state):
        token = state["token_of_interest"]
        chain = state["chain"]
        token_context = build_token_context(token, chain)

        tools = [
            get_token_metadata,
            get_onchain_metrics,
        ]

        system_message = (
            "You are a crypto On-Chain Analyst. Your job is to evaluate the "
            "fundamental safety and health of a token using on-chain data.\n\n"
            "Analyze these dimensions:\n"
            "1. **Contract Security**: Is the contract verified? Is it a proxy "
            "(upgradeable)? Does it have mint/blacklist functions? Any honeypot risk?\n"
            "2. **Holder Analysis**: How concentrated is the token? What do the top "
            "holders look like? Are LP tokens locked?\n"
            "3. **Liquidity Quality**: How deep is liquidity? Is it locked? How long?\n"
            "4. **Tax Structure**: What are the buy/sell taxes? Are they reasonable?\n"
            "5. **Age & Maturity**: How old is the token? Early-stage (<24h) tokens "
            "carry dramatically higher risk.\n\n"
            "Score each dimension and provide an overall security rating. "
            "Flag any CRITICAL or HIGH risk issues prominently.\n\n"
            "Use the tools to fetch token metadata and on-chain security data.\n\n"
            "End with a Markdown table summarizing your findings."
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
            "onchain_report": report,
        }

    return onchain_analyst_node
