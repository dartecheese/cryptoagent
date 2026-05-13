"""Conservative Risk Debator — argues against the trade, stress-tests every assumption.

Focuses on worst-case scenarios, hidden risks, and capital preservation.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from cryptoagent.agents.utils.agent_utils import get_language_instruction


def create_conservative_debator(llm):

    def conservative_debator_node(state):
        risk_state = state["risk_debate_state"]
        trader_plan = state.get("trader_investment_plan", "No trader plan available.")
        research_plan = state.get("investment_plan", "")
        token = state["token_of_interest"]
        chain = state["chain"]

        system_message = (
            f"You are a Conservative Risk Analyst for a crypto trading desk.\n\n"
            f"Token: `{token}` on `{chain}`\n\n"
            "Your bias: capital preservation comes first. Crypto is full of ways to "
            "lose money. Your job is to find every reason NOT to take this trade.\n\n"
            "Stress-test this trade:\n"
            "1. **Worst case**: What if this goes to zero? How much do we lose?\n"
            "2. **Exit risk**: Can we actually exit this position? What's the liquidity "
            "depth? Is there a real buyer at these levels?\n"
            "3. **Hidden risks**: Is there a proxy contract? Can the owner mint? "
            "Can LP be pulled? Are there pending unlocks?\n"
            "4. **MEV vulnerability**: Will this trade get sandwiched? What does "
            "the mempool look like?\n"
            "5. **Correlation risk**: If our other positions are also DeFi tokens, "
            "are we over-concentrated?\n"
            "6. **Time risk**: Memecoins can go -50% in 5 minutes. Does this position "
            "have a real stop loss or just a hope?\n\n"
            "Previous debate:\n{debate_history}\n\n"
            "Trader's Plan:\n{trader_plan}\n\n"
            "Research Plan:\n{research_plan}"
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ])

        prompt = prompt.partial(
            trader_plan=trader_plan,
            research_plan=research_plan,
            debate_history=risk_state.get("history", "No prior debate."),
        )

        chain = prompt | llm
        result = chain.invoke(state["messages"])

        return {
            "messages": [result],
            "risk_debate_state": {
                **risk_state,
                "conservative_history": risk_state.get("conservative_history", "") + "\n" + result.content,
                "history": risk_state.get("history", "") + "\n[CONSERVATIVE]: " + result.content,
                "current_conservative_response": result.content,
                "latest_speaker": "conservative",
                "count": risk_state.get("count", 0) + 1,
            },
        }

    return conservative_debator_node
