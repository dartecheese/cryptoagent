"""Shared base for crypto analysts — ultra-compact prompts for token efficiency."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def make_analyst_node(llm, tools: list, system_message: str, report_key: str):
    """Factory: create a LangGraph analyst node with compact prompt.

    Injects token + chain context so the LLM knows what to analyze.
    """
    def node(state):
        token = state.get("token_of_interest", state.get("company_of_interest", "unknown"))
        chain = state.get("chain", "ethereum")

        full_system = (
            f"Token: {token} on {chain}. "
            "Use tools to fetch data. Output ONLY valid JSON. No explanation, just JSON.\n"
            + system_message
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", full_system),
            MessagesPlaceholder(variable_name="messages"),
        ])
        chain_prompt = prompt | llm.bind_tools(tools)
        result = chain_prompt.invoke({"messages": state["messages"]})

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            report_key: report,
        }
    return node
