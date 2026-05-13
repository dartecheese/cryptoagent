"""Narrative Analyst — identifies which meta-narratives are driving the market.

NEW crypto-specific agent. Analyzes:
- Which narratives are currently trending (memecoins, AI agents, L2s, RWA, etc.)
- Narrative momentum: are we early, at peak, or post-peak?
- Token's alignment with hot narratives
- Rotation signals (capital flowing from one narrative to another)

Data: DexScreener trending/boosts, CoinGecko trending.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from cryptoagent.agents.utils.agent_utils import (
    build_token_context,
    get_language_instruction,
    get_narrative_data,
    get_token_metadata,
)


def create_narrative_analyst(llm):
    """Create the narrative analyst node."""

    def narrative_analyst_node(state):
        token = state["token_of_interest"]
        chain = state["chain"]
        token_context = build_token_context(token, chain)

        tools = [
            get_narrative_data,
            get_token_metadata,
        ]

        system_message = (
            "You are a crypto Narrative Analyst. Your job is to identify which "
            "market narratives are driving attention and capital flows, and assess "
            "how well a given token aligns with those narratives.\n\n"
            "Analyze these dimensions:\n"
            "1. **Active Narratives**: What categories are trending right now? "
            "(memecoins, AI agents, RWA, DePIN, L2s, gaming, etc.)\n"
            "2. **Narrative Momentum**: Is this narrative early (growing), at peak "
            "(saturated), or exhausted (capital rotating out)?\n"
            "3. **Token Alignment**: How well does this specific token fit the active "
            "narratives? Is it a leader or a follower?\n"
            "4. **Rotation Signals**: Are you seeing capital rotate FROM one narrative "
            "TO another? This is critical for timing.\n"
            "5. **Boost Activity**: Are tokens in this category being boosted on "
            "DexScreener? Boosts = paid promotion = narrative signal.\n\n"
            "Provide a clear NARRATIVE RATING: 🟢 Strong Alignment / 🟡 Neutral / "
            "🔴 Misaligned. End with a Markdown table."
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
            "narrative_report": report,
        }

    return narrative_analyst_node
