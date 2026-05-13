"""Conditional routing logic for the CryptoAgent LangGraph graph.

Mirrors TradingAgents' conditional_logic.py — controls when to continue
tool calls, when to advance debate rounds, and when to move between agents.
"""


class ConditionalLogic:
    """Handles all conditional routing decisions in the graph."""

    def __init__(
        self,
        max_debate_rounds: int = 2,
        max_risk_discuss_rounds: int = 2,
    ):
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def _should_continue_tools(self, state, tools_key: str):
        """Generic tool continuation check for analyst nodes."""
        messages = state["messages"]
        last_message = messages[-1] if messages else None

        if last_message and hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
            return tools_key
        return "clear"

    def should_continue_onchain(self, state):
        return self._should_continue_tools(state, "tools_onchain")

    def should_continue_sentiment(self, state):
        return self._should_continue_tools(state, "tools_sentiment")

    def should_continue_narrative(self, state):
        return self._should_continue_tools(state, "tools_narrative")

    def should_continue_technical(self, state):
        return self._should_continue_tools(state, "tools_technical")

    def should_continue_debate(self, state):
        """Continue bull/bear debate until max rounds or Research Manager intervenes."""
        debate_state = state.get("investment_debate_state", {})
        count = debate_state.get("count", 0)

        if count >= self.max_debate_rounds * 2:
            return "Research Manager"

        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        if last_message and hasattr(last_message, "content"):
            content = str(last_message.content)
            if "FINAL TRANSACTION PROPOSAL" in content:
                return "Research Manager"

        # Alternate: bull → bear → bull → bear → ...
        last_speaker = None
        history = debate_state.get("history", "")
        if "[BULL]" in history and "[BEAR]" not in history.split("[BULL]")[-1]:
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state):
        """Continue risk debate until max rounds or Portfolio Manager steps in."""
        risk_state = state.get("risk_debate_state", {})
        count = risk_state.get("count", 0)

        if count >= self.max_risk_discuss_rounds * 3:
            return "Portfolio Manager"

        latest = risk_state.get("latest_speaker", "")
        # Rotate: aggressive → conservative → neutral → aggressive → ...
        rotation = {"aggressive": "Conservative Analyst",
                     "conservative": "Neutral Analyst",
                     "neutral": "Aggressive Analyst"}
        return rotation.get(latest, "Conservative Analyst")
