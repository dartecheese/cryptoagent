"""Neutral Risk Debator — balanced assessment weighing both sides.

Provides the swing vote in the risk debate — not biased toward action or inaction.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from cryptoagent.agents.utils.agent_utils import get_language_instruction


def create_neutral_debator(llm):

    def neutral_debator_node(state):
        risk_state = state["risk_debate_state"]
        trader_plan = state.get("trader_investment_plan", "No trader plan available.")
        research_plan = state.get("investment_plan", "")
        token = state["token_of_interest"]
        chain = state["chain"]

        system_message = (
            f"You are a Neutral Risk Analyst for a crypto trading desk.\n\n"
            f"Token: `{token}` on `{chain}`\n\n"
            "Your bias: you weigh both sides objectively. You're not trying to be "
            "contrarian — you're trying to find the truth.\n\n"
            "Analyze this trade by:\n"
            "1. Separating signal from noise in both the aggressive and conservative arguments\n"
            "2. Identifying what BOTH sides are missing\n"
            "3. Quantifying the risk/reward ratio: is it above 2:1? 3:1?\n"
            "4. Assessing whether position sizing is appropriate for the risk level\n"
            "5. Giving a clear lean — you're neutral, not indecisive. Pick a direction.\n\n"
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
                "neutral_history": risk_state.get("neutral_history", "") + "\n" + result.content,
                "history": risk_state.get("history", "") + "\n[NEUTRAL]: " + result.content,
                "current_neutral_response": result.content,
                "latest_speaker": "neutral",
                "count": risk_state.get("count", 0) + 1,
            },
        }

    return neutral_debator_node
