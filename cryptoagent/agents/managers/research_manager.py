"""Research Manager — synthesizes the bull/bear debate into an investment plan.

Uses structured output (ResearchPlan Pydantic schema) for consistent, parseable decisions.
The deep-thinking LLM is used here because this is the highest-stakes synthesis step.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from cryptoagent.agents.utils.agent_utils import get_language_instruction
from cryptoagent.agents.schemas import ResearchPlan, render_research_plan


def create_research_manager(llm, provider: str = "openai"):
    """Create the research manager node — uses structured output."""

    # DeepSeek doesn't support json_schema — use function_calling instead
    method = "function_calling" if provider == "deepseek" else "json_schema"
    structured_llm = llm.with_structured_output(ResearchPlan, method=method)

    def research_manager_node(state):
        debate_state = state["investment_debate_state"]
        token = state["token_of_interest"]
        chain = state["chain"]

        system_message = (
            "You are the Research Manager for a crypto trading desk. Your job is to "
            "review the bull and bear debate and produce a clear investment plan.\n\n"
            f"Token: `{token}` on `{chain}`\n\n"
            "Guidelines:\n"
            "- **Buy**: Strong evidence across multiple analysts, bull case clearly wins\n"
            "- **Overweight**: Positive tilt, but some concerns remain\n"
            "- **Hold**: Evidence is genuinely balanced, or too many unknowns\n"
            "- **Underweight**: Bearish tilt, but not catastrophic\n"
            "- **Sell**: Clear danger signals — rug risk, exploited contract, exit liquidity trap\n\n"
            "Consider crypto-specific factors:\n"
            "- Token age: < 24h old = HIGH RISK regardless of other signals\n"
            "- Liquidity: < $50K liquidity = dangerous to enter\n"
            "- Holder concentration: > 60% top 10 = whale exit risk\n"
            "- Narrative timing: peak narrative = greater fool risk\n\n"
            "Bull arguments:\n{bull}\n\n"
            "Bear arguments:\n{bear}\n\n"
            "Produce a structured investment plan with recommendation, rationale, "
            "strategic actions, and conviction level."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ])

        prompt = prompt.partial(
            bull=debate_state.get("bull_history", "No bull case presented."),
            bear=debate_state.get("bear_history", "No bear case presented."),
        )

        chain = prompt | structured_llm
        plan: ResearchPlan = chain.invoke(state["messages"])
        rendered = render_research_plan(plan)

        return {
            "messages": [rendered],
            "investment_plan": rendered,
            "investment_debate_state": {
                **debate_state,
                "judge_decision": plan.recommendation.value,
            },
        }

    return research_manager_node
