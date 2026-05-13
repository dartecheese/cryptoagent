"""Bear Researcher — builds the strongest bearish case from analyst reports."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from cryptoagent.agents.utils.agent_utils import get_language_instruction


def create_bear_researcher(llm):
    """Create the bear researcher node."""

    def bear_researcher_node(state):
        debate_state = state["investment_debate_state"]
        token = state["token_of_interest"]
        chain = state["chain"]

        reports = []
        if state.get("onchain_report"):
            reports.append(f"## On-Chain Analysis\n{state['onchain_report']}")
        if state.get("sentiment_report"):
            reports.append(f"## Sentiment Analysis\n{state['sentiment_report']}")
        if state.get("narrative_report"):
            reports.append(f"## Narrative Analysis\n{state['narrative_report']}")
        if state.get("technical_report"):
            reports.append(f"## Technical Analysis\n{state['technical_report']}")

        all_reports = "\n\n".join(reports)

        system_message = (
            "You are a Bearish Crypto Researcher. Your role is to build the strongest "
            "possible BEAR case against trading this token.\n\n"
            f"Token: `{token}` on `{chain}`\n\n"
            "You have access to four analyst reports (on-chain, sentiment, narrative, "
            "technical). Your job is to:\n"
            "1. Extract every bearish data point and risk factor from each report\n"
            "2. Build the most compelling case for why this trade could fail\n"
            "3. Identify specific risks: rug potential, liquidity trap, narrative "
            "exhaustion, whale distribution, MEV vulnerability\n"
            "4. Challenge the bull case directly — if the bull researcher made arguments, "
            "find the weaknesses\n"
            "5. Be intellectually honest — don't be bearish just to be bearish, "
            "but don't pull punches either\n\n"
            "Previous debate:\n{debate_history}\n\n"
            "Analyst Reports:\n{reports}"
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ])

        prompt = prompt.partial(
            reports=all_reports,
            debate_history=debate_state.get("history", "No prior debate."),
        )

        chain = prompt | llm
        result = chain.invoke(state["messages"])

        return {
            "messages": [result],
            "investment_debate_state": {
                **debate_state,
                "bear_history": debate_state.get("bear_history", "") + "\n" + result.content,
                "history": debate_state.get("history", "") + "\n[BEAR]: " + result.content,
                "current_response": result.content,
                "count": debate_state.get("count", 0) + 1,
            },
        }

    return bear_researcher_node
