"""Aggressive Risk Debator — argues for taking the trade with maximum conviction.

Focuses on upside potential, manageable risks, and opportunity cost of not trading.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from cryptoagent.agents.utils.agent_utils import get_language_instruction


def create_aggressive_debator(llm):

    def aggressive_debator_node(state):
        risk_state = state["risk_debate_state"]
        trader_plan = state.get("trader_investment_plan", "No trader plan available.")
        research_plan = state.get("investment_plan", "")
        token = state["token_of_interest"]
        chain = state["chain"]

        system_message = (
            f"You are an Aggressive Risk Analyst for a crypto trading desk.\n\n"
            f"Token: `{token}` on `{chain}`\n\n"
            "Your bias: you believe the best opportunities come from taking calculated "
            "risks. Crypto rewards conviction. Being too cautious means missing 100x.\n\n"
            "Argue for taking this trade. Focus on:\n"
            "1. The asymmetric upside — if this works, what's the return profile?\n"
            "2. Why the risks are manageable at this position size\n"
            "3. The opportunity cost of NOT trading (FOMO is real in crypto)\n"
            "4. Convexity: the most you can lose is your position, but the upside "
            "is unbounded (especially in memecoins)\n"
            "5. Address the conservative concerns preemptively\n\n"
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
                "aggressive_history": risk_state.get("aggressive_history", "") + "\n" + result.content,
                "history": risk_state.get("history", "") + "\n[AGGRESSIVE]: " + result.content,
                "current_aggressive_response": result.content,
                "latest_speaker": "aggressive",
                "count": risk_state.get("count", 0) + 1,
            },
        }

    return aggressive_debator_node
