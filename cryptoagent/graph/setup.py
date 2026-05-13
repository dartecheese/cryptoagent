"""Graph setup — constructs the LangGraph workflow.

Crypto-adapted from TradingAgents' setup.py. The graph follows this flow:
START → On-Chain Analyst → Sentiment Analyst → Narrative Analyst → Technical Analyst
     → Bull Researcher ⇄ Bear Researcher (debate)
     → Research Manager → Trader
     → Aggressive ⇄ Conservative ⇄ Neutral (risk debate)
     → Portfolio Manager → END
"""

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from cryptoagent.agents import *
from cryptoagent.agents.utils.agent_states import AgentState
from cryptoagent.graph.conditional_logic import ConditionalLogic


class GraphSetup:
    """Builds and configures the LangGraph agent workflow."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        provider: str = "openai",
    ):
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.provider = provider

    def setup_graph(
        self,
        selected_analysts: list[str] | None = None,
    ):
        """Set up the full agent workflow graph.

        Args:
            selected_analysts: Which analysts to include.
                Default: all four ['onchain', 'sentiment', 'narrative', 'technical']
        """
        if selected_analysts is None:
            selected_analysts = ["onchain", "sentiment", "narrative", "technical"]

        if len(selected_analysts) == 0:
            raise ValueError("At least one analyst must be selected!")

        # ── Create Analyst Nodes ────────────────────────
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        analyst_factories = {
            "onchain": (create_onchain_analyst, "onchain"),
            "sentiment": (create_sentiment_analyst, "sentiment"),
            "narrative": (create_narrative_analyst, "narrative"),
            "technical": (create_technical_analyst, "technical"),
        }

        for key in selected_analysts:
            if key not in analyst_factories:
                continue
            factory, tool_key = analyst_factories[key]
            analyst_nodes[key] = factory(self.quick_thinking_llm)
            delete_nodes[key] = create_msg_delete()
            tool_nodes[key] = self.tool_nodes[tool_key]

        # ── Create Researcher & Manager Nodes ───────────
        bull_researcher_node = create_bull_researcher(self.quick_thinking_llm)
        bear_researcher_node = create_bear_researcher(self.quick_thinking_llm)
        research_manager_node = create_research_manager(self.deep_thinking_llm, self.provider)
        trader_node = create_trader(self.quick_thinking_llm, self.provider)

        # ── Create Risk Nodes ───────────────────────────
        aggressive_node = create_aggressive_debator(self.quick_thinking_llm)
        neutral_node = create_neutral_debator(self.quick_thinking_llm)
        conservative_node = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm, self.provider)

        # ── Build Workflow ──────────────────────────────
        workflow = StateGraph(AgentState)

        # Formatting: "Onchain Analyst", "Sentiment Analyst", etc.
        def _title(key: str) -> str:
            return f"{key.capitalize()} Analyst"

        def _clear(key: str) -> str:
            return f"Msg Clear {key.capitalize()}"

        def _tools(key: str) -> str:
            return f"tools_{key}"

        # Add analyst nodes + clear nodes + tools
        for key in selected_analysts:
            if key not in analyst_nodes:
                continue
            workflow.add_node(_title(key), analyst_nodes[key])
            workflow.add_node(_clear(key), delete_nodes[key])
            workflow.add_node(_tools(key), tool_nodes[key])

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_node)
        workflow.add_node("Neutral Analyst", neutral_node)
        workflow.add_node("Conservative Analyst", conservative_node)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # ── Wire Edges ──────────────────────────────────
        # Start → first analyst
        first_key = selected_analysts[0]
        workflow.add_edge(START, _title(first_key))

        # Connect analysts in sequence with tool loops
        for i, key in enumerate(selected_analysts):
            if key not in analyst_nodes:
                continue

            # Analyst → tools or clear
            should_continue = getattr(
                self.conditional_logic, f"should_continue_{key}"
            )
            workflow.add_conditional_edges(
                _title(key),
                should_continue,
                {_tools(key): _tools(key), "clear": _clear(key)},
            )
            workflow.add_edge(_tools(key), _title(key))

            # Clear → next analyst or Bull Researcher
            if i < len(selected_analysts) - 1:
                next_key = selected_analysts[i + 1]
                workflow.add_edge(_clear(key), _title(next_key))
            else:
                workflow.add_edge(_clear(key), "Bull Researcher")

        # Bull/Bear debate loop
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )

        # Research Manager → Trader
        workflow.add_edge("Research Manager", "Trader")

        # Trader → Risk debate
        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        # Portfolio Manager → END
        workflow.add_edge("Portfolio Manager", END)

        return workflow
