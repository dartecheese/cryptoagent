"""Portfolio Manager — final decision gate before execution.

Uses structured output (PortfolioDecision) with crypto-specific fields:
risk_flags, narrative_tags, time_horizon. The deep-thinking LLM is used here
because this is the final go/no-go decision with capital at stake.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from cryptoagent.agents.utils.agent_utils import get_language_instruction
from cryptoagent.agents.schemas import PortfolioDecision, render_pm_decision


def create_portfolio_manager(llm, provider: str = "openai"):
    """Create the portfolio manager node — uses structured output."""

    method = "function_calling" if provider == "deepseek" else "json_schema"
    structured_llm = llm.with_structured_output(PortfolioDecision, method=method)

    def portfolio_manager_node(state):
        risk_state = state["risk_debate_state"]
        trader_plan = state.get("trader_investment_plan", "")
        research_plan = state.get("investment_plan", "")
        past_context = state.get("past_context", "")
        token = state["token_of_interest"]
        chain = state["chain"]

        system_message = (
            "You are the Portfolio Manager for a crypto trading desk. You have the "
            "final say on every trade. Capital preservation is your #1 priority, "
            "but you must also capture asymmetric upside when the thesis is strong.\n\n"
            f"Token: `{token}` on `{chain}`\n\n"
            "You must decide: do we execute this trade or pass?\n\n"
            "Criteria (crypto-specific):\n"
            "- **On-chain safety**: If any CRITICAL security risk exists, vote SELL/UNDERWEIGHT\n"
            "- **Liquidity**: Can we enter and exit without moving the price >3%?\n"
            "- **Narrative timing**: Are we early, at peak, or post-peak?\n"
            "- **Position sizing**: Is the size appropriate for the risk level?\n"
            "- **Portfolio context**: How does this fit with existing positions?\n"
            "- **Past lessons**: What have we learned from similar trades?\n\n"
            "Risk Debate:\n"
            f"Aggressive: {risk_state.get('current_aggressive_response', 'No input.')}\n"
            f"Conservative: {risk_state.get('current_conservative_response', 'No input.')}\n"
            f"Neutral: {risk_state.get('current_neutral_response', 'No input.')}\n\n"
            f"Trader's Plan:\n{trader_plan}\n\n"
            f"Research Plan:\n{research_plan}\n\n"
            f"Past Trading Lessons:\n{past_context if past_context else 'No prior lessons.'}"
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ])

        chain = prompt | structured_llm
        decision: PortfolioDecision = chain.invoke(state["messages"])
        rendered = render_pm_decision(decision)

        return {
            "messages": [rendered],
            "final_trade_decision": rendered,
            "risk_debate_state": {
                **risk_state,
                "judge_decision": decision.rating.value,
            },
        }

    return portfolio_manager_node
