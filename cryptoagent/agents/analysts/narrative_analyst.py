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
            "You are a crypto Narrative Analyst. Identify market meta concisely.\n\n"
            "REQUIRED OUTPUT FORMAT (terse):\n"
            "1. Active Narratives: [list top 3 trending categories]\n"
            "2. Token Alignment: [STRONG|WEAK|MISALIGNED] — why\n"
            "3. Narrative Phase: [EARLY|GROWING|PEAK|EXHAUSTED]\n"
            "4. Rotation Signal: [yes/no] — capital flowing in or out?\n\n"
            "END WITH EXACTLY:\n"
            "NARRATIVE: [STRONG_ALIGNMENT|NEUTRAL|MISALIGNED]\n"
            "MOMENTUM: [BUILDING|STEADY|FADING]"
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
