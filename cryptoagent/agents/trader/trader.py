"""Crypto Trader — translates the research plan into concrete DeFi parameters.

Uses structured output (TraderProposal) with crypto-specific fields:
chain, entry_type, slippage, stop loss, take profit, position size.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from cryptoagent.agents.utils.agent_utils import get_language_instruction
from cryptoagent.agents.schemas import TraderProposal, render_trader_proposal


def create_trader(llm, provider: str = "openai"):
    """Create the crypto trader node — uses structured output."""

    method = "function_calling" if provider == "deepseek" else "json_schema"
    structured_llm = llm.with_structured_output(TraderProposal, method=method)

    def trader_node(state):
        token = state["token_of_interest"]
        chain = state["chain"]
        investment_plan = state.get("investment_plan", "No investment plan available.")

        system_message = (
            "You are a Crypto Trader. Your job is to take the Research Manager's "
            "investment plan and translate it into concrete execution parameters "
            "for DeFi trading.\n\n"
            f"Token: `{token}` on `{chain}`\n\n"
            "You must specify:\n"
            "- **Action**: Buy, Sell, or Hold\n"
            "- **Chain**: which blockchain to execute on\n"
            "- **Entry Type**: Market (immediate, higher slippage), Limit (set price, "
            "may not fill), or TWAP (time-weighted, minimizes impact)\n"
            "- **Entry Price**: target entry price in USD if using Limit\n"
            "- **Stop Loss**: percentage below entry (e.g. 0.15 = 15%). Typical: "
            "5-15% for swing, 2-5% for scalp\n"
            "- **Take Profit**: percentage above entry (e.g. 1.0 = 100%). Memecoins "
            "often need higher TP, blue chips lower\n"
            "- **Position Size**: in USD. Never exceed 25% of portfolio on a single trade. "
            "For new tokens (<24h), max 5%\n"
            "- **Max Slippage**: in basis points (300 = 3%). Higher for memecoins, "
            "lower for stable pairs\n\n"
            "Crypto-specific wisdom:\n"
            "- MEV is real: market orders on DEXs get sandwiched. Use high slippage "
            "tolerance or private mempools\n"
            "- Gas matters: if position < $100, gas might eat your profits\n"
            "- Exit liquidity: in low-liquidity pools, your exit IS the exit event\n\n"
            "Investment Plan:\n{investment_plan}"
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ])

        prompt = prompt.partial(investment_plan=investment_plan)
        chain = prompt | structured_llm
        proposal: TraderProposal = chain.invoke(state["messages"])
        rendered = render_trader_proposal(proposal)

        return {
            "messages": [rendered],
            "trader_investment_plan": rendered,
        }

    return trader_node
