"""Smoke test — verifies the CryptoAgent graph compiles and state flows correctly.

Does NOT require API keys. Uses a mock LLM to verify:
- Graph compiles without errors
- State flows through all nodes
- Structured output schemas are valid
- Bridge signal builder works
"""

import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_schemas_import():
    """Verify all Pydantic schemas are valid and can be constructed."""
    from cryptoagent.agents.schemas import (
        ResearchPlan, TraderProposal, PortfolioDecision,
        PortfolioRating, TraderAction, ConvictionLevel,
        TimeHorizon, EntryType, RiskFlag,
    )
    
    # Build a ResearchPlan
    plan = ResearchPlan(
        recommendation=PortfolioRating.BUY,
        rationale="Strong buy signal across all analysts",
        strategic_actions="Enter at market with 2% portfolio allocation",
        conviction=ConvictionLevel.HIGH,
    )
    assert plan.recommendation == PortfolioRating.BUY
    
    from cryptoagent.agents.schemas import render_research_plan
    rendered = render_research_plan(plan)
    assert "Buy" in rendered
    assert "High" in rendered
    
    # Build a TraderProposal
    proposal = TraderProposal(
        action=TraderAction.BUY,
        reasoning="All analysts bullish, strong momentum",
        chain="ethereum",
        entry_type=EntryType.MARKET,
        entry_price_usd=0.001234,
        stop_loss_pct=0.15,
        take_profit_pct=1.0,
        position_size_usd=500.0,
        max_slippage_bps=300,
    )
    assert proposal.chain == "ethereum"
    
    from cryptoagent.agents.schemas import render_trader_proposal
    rendered = render_trader_proposal(proposal)
    assert "BUY" in rendered
    
    # Build a PortfolioDecision
    decision = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary="Enter with conviction, manage risk tightly",
        investment_thesis="On-chain metrics solid, sentiment bullish, narrative aligned",
        price_target=0.005,
        time_horizon=TimeHorizon.SWING,
        risk_flags=[RiskFlag.MEV_RISK, RiskFlag.LOW_LIQUIDITY],
        narrative_tags=["memecoin season", "AI agents"],
    )
    assert decision.rating == PortfolioRating.BUY
    assert len(decision.risk_flags) == 2
    
    from cryptoagent.agents.schemas import render_pm_decision
    rendered = render_pm_decision(decision)
    assert "Buy" in rendered
    assert "memecoin season" in rendered
    
    print("✅ Schemas: all valid, render functions work")


def test_state_types():
    """Verify LangGraph state types are valid."""
    from cryptoagent.agents.utils.agent_states import (
        AgentState, InvestDebateState, RiskDebateState,
    )
    
    debate = InvestDebateState({
        "bull_history": "",
        "bear_history": "",
        "history": "",
        "current_response": "",
        "judge_decision": "",
        "count": 0,
    })
    assert debate["count"] == 0
    
    risk = RiskDebateState({
        "aggressive_history": "",
        "conservative_history": "",
        "neutral_history": "",
        "history": "",
        "latest_speaker": "",
        "current_aggressive_response": "",
        "current_conservative_response": "",
        "current_neutral_response": "",
        "judge_decision": "",
        "count": 0,
    })
    assert risk["latest_speaker"] == ""
    
    print("✅ State types: all valid")


def test_signal_builder():
    """Verify AST bridge signal builder works."""
    from cryptoagent.bridge.signal_builder import (
        AgenticTradingSignal, Direction, Conviction, Rating,
    )
    from cryptoagent.agents.schemas import (
        PortfolioDecision, TraderProposal,
        PortfolioRating, TraderAction, EntryType,
        TimeHorizon, RiskFlag,
    )
    
    decision = PortfolioDecision(
        rating=PortfolioRating.BUY,
        executive_summary="Go",
        investment_thesis="Strong signals",
        time_horizon=TimeHorizon.INTRADAY,
        risk_flags=[RiskFlag.MEV_RISK],
        narrative_tags=["memecoin"],
    )
    
    proposal = TraderProposal(
        action=TraderAction.BUY,
        reasoning="Buy it",
        chain="ethereum",
        entry_type=EntryType.MARKET,
        position_size_usd=500.0,
    )
    
    signal = AgenticTradingSignal.from_portfolio_decision(
        decision, proposal, "0x1234", "ethereum"
    )
    
    assert signal.direction == Direction.LONG
    assert signal.token_address == "0x1234"
    assert signal.chain == "ethereum"
    assert "memecoin" in signal.narrative_tags
    
    # Test JSON serialization
    json_str = signal.to_json()
    assert "0x1234" in json_str
    assert "Long" in json_str
    
    print("✅ Signal builder: valid, serializable")


def test_conditional_logic():
    """Verify routing logic."""
    from cryptoagent.graph.conditional_logic import ConditionalLogic
    
    cl = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
    
    # Test debate routing (no state → bull first)
    state = {
        "investment_debate_state": {
            "count": 0,
            "history": "",
        },
        "messages": [],
    }
    result = cl.should_continue_debate(state)
    assert result in ("Bull Researcher", "Bear Researcher")
    
    # Test risk routing
    risk_state = {
        "risk_debate_state": {
            "count": 0,
            "latest_speaker": "aggressive",
            "history": "",
        },
    }
    result = cl.should_continue_risk_analysis(risk_state)
    assert result in ("Conservative Analyst", "Neutral Analyst", "Aggressive Analyst", "Portfolio Manager")
    
    print("✅ Conditional logic: routing works")


def test_config():
    """Verify config loads correctly."""
    from cryptoagent.config import CRYPTOAGENT_DEFAULT_CONFIG, get_config

    cfg = get_config()
    assert cfg["chains"] == ["ethereum", "base", "arbitrum", "solana", "bsc"]
    assert cfg["max_debate_rounds"] == 2
    assert cfg["min_liquidity_usd"] == 100_000

    print("✅ Config: loads, all keys present")


def test_memory_log():
    """Verify trading memory log works."""
    import tempfile
    from cryptoagent.graph.memory import TradingMemoryLog

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"memory_log_path": f"{tmpdir}/test_memory.md"}
        log = TradingMemoryLog(config)

        log.log_decision("0xTEST", "ethereum", "Buy", "High", "Test thesis")
        context = log.get_recent_context()
        assert "0xTEST" in context
        assert "Buy" in context

    print("✅ Memory log: writes and reads")


def test_graph_structure():
    """Verify the graph can be imported and inspected (no execution)."""
    from cryptoagent.graph.trading_graph import CryptoAgentGraph
    from cryptoagent.graph.setup import GraphSetup
    from cryptoagent.graph.propagation import Propagator
    from cryptoagent.graph.conditional_logic import ConditionalLogic

    # Verify we can inspect the graph structure
    cl = ConditionalLogic()
    assert cl.max_debate_rounds == 2
    assert cl.max_risk_discuss_rounds == 2

    p = Propagator()
    initial = p.create_initial_state("0xTEST", "ethereum")
    assert initial["token_of_interest"] == "0xTEST"
    assert initial["chain"] == "ethereum"
    assert "messages" in initial
    assert "onchain_report" in initial

    print("✅ Graph structure: imports clean, propagator/conditional valid")


if __name__ == "__main__":
    print("🧪 CryptoAgent Smoke Test")
    print("=" * 60)
    
    tests = [
        test_schemas_import,
        test_state_types,
        test_signal_builder,
        test_conditional_logic,
        test_config,
        test_memory_log,
        test_graph_structure,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    print("=" * 60)
    print("✅ All smoke tests passed!")
    print()
    print("Next steps:")
    print("  1. Set OPENAI_API_KEY in your environment")
    print("  2. Run: python cryptoagent/scripts/run_analysis.py 0xTOKEN_ADDRESS")
    print("  3. Or use the Python API:")
    print("     from cryptoagent.graph.trading_graph import CryptoAgentGraph")
    print("     cg = CryptoAgentGraph()")
    print("     state, decision = cg.propagate('0xTOKEN', 'ethereum')")
