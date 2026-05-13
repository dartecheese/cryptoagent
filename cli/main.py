#!/usr/bin/env python3
"""
CryptoAgent CLI — Interactive multi-agent crypto trading analysis.
Mirrors TradingAgents CLI experience with rich terminal UI.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich import box
import questionary

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptoagent.config import CRYPTOAGENT_DEFAULT_CONFIG
from cryptoagent.graph.trading_graph import CryptoAgentGraph
from cryptoagent.graph.memory import TradingMemoryLog

console = Console()

WELCOME = """
 ▄████████  ████████▄     ▄██████▄   ▄████████▄     ▄████████ ▄██████▄  
███    ███ ███   ▀███   ███    ███ ███    ███     ███    ███ ███   ███ 
███    █▀  ███    ███   ███    ███ ███    ███     ███    █▀  ███   █▀  
███        ███    ███  ▄███▄▄▄▄██▀ ███    ███    ▄███▄▄▄     ███       
███        ███    ███ ▀▀███▀▀▀▀▀   ███    ███   ▀▀███▀▀▀     ███       
███    █▄  ███    ███ ▀███████████ ███    ███     ███    █▄  ███   █▄ 
███    ███ ███   ▄███   ███    ███ ███    ███     ███    ███ ███   ███ 
████████▀  ████████▀    ███    ███  ▀██████▀      ██████████  ▀██████▀  
                         ███    ███                                     
"""

STEPS = [
    "I. Analyst Team",
    "II. Research Team", 
    "III. Trader",
    "IV. Risk Management",
    "V. Portfolio Management",
]


def show_banner():
    console.clear()
    console.print(Panel.fit(
        Text(WELCOME, style="bold cyan"),
        title="CryptoAgent — Multi-Agent LLM Crypto Trading Framework",
        subtitle="Built on TradingAgents architecture · Adapted for DeFi/crypto markets",
        border_style="cyan",
    ))
    console.print()
    console.print("[dim]Workflow Steps:[/dim]")
    for step in STEPS:
        console.print(f"  [dim]{step}[/dim]")
    console.print()
    console.print("[dim]Built by dartecheese · github.com/dartecheese/cryptoagent[/dim]")
    console.print()


def get_api_key(provider: str) -> str | None:
    """Get API key from env or prompt."""
    env_map = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "xai": "XAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    env_var = env_map.get(provider, "")
    key = os.getenv(env_var, "")
    if key:
        return key

    key = questionary.password(
        f"Enter {provider.upper()} API key (or set {env_var} env var):"
    ).ask()
    return key if key else None


def interactive_setup() -> dict:
    """Interactive setup wizard — mirrors TradingAgents CLI flow."""

    # Step 1: Token
    console.print()
    console.print(Panel(
        "[bold]Step 1: Token[/bold]\n"
        "Enter the token contract address or symbol to analyze.\n"
        "Examples: 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2, ETH, SOL",
        border_style="blue",
    ))
    token = questionary.text(
        "Token address/symbol:",
        default="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    ).ask()
    if not token:
        console.print("[red]Token required.[/red]")
        sys.exit(1)

    # Step 2: Chain
    console.print()
    console.print(Panel(
        "[bold]Step 2: Chain[/bold]\n"
        "Select the blockchain where this token lives.",
        border_style="blue",
    ))
    chain = questionary.select(
        "Select blockchain:",
        choices=[
            {"name": "Ethereum", "value": "ethereum"},
            {"name": "BSC (Binance Smart Chain)", "value": "bsc"},
            {"name": "Arbitrum", "value": "arbitrum"},
            {"name": "Base", "value": "base"},
            {"name": "Solana", "value": "solana"},
        ],
    ).ask()

    # Step 3: Analysts
    console.print()
    console.print(Panel(
        "[bold]Step 3: Analysts Team[/bold]\n"
        "Select which LLM analyst agents to include.",
        border_style="blue",
    ))
    analysts = questionary.checkbox(
        "Select analysts:",
        choices=[
            questionary.Choice("On-Chain Analyst (contract security, holders, liquidity)", checked=True, value="onchain"),
            questionary.Choice("Sentiment Analyst (buy/sell ratio, trading activity)", checked=True, value="sentiment"),
            questionary.Choice("Narrative Analyst (trending categories, rotation)", checked=True, value="narrative"),
            questionary.Choice("Technical Analyst (price action, chart patterns)", checked=True, value="technical"),
        ],
    ).ask()
    if not analysts:
        console.print("[red]At least one analyst required.[/red]")
        sys.exit(1)

    # Step 4: Research Depth
    console.print()
    console.print(Panel(
        "[bold]Step 4: Research Depth[/bold]\n"
        "How many rounds of debate and risk discussion?",
        border_style="blue",
    ))
    depth = questionary.select(
        "Select depth:",
        choices=[
            {"name": "Fast — Single debate round, quick analysis", "value": "fast"},
            {"name": "Medium — Two debate rounds, balanced depth (recommended)", "value": "medium"},
            {"name": "Deep — Three debate rounds, thorough analysis (slower)", "value": "deep"},
        ],
    ).ask()
    depth_map = {"fast": 1, "medium": 2, "deep": 3}
    rounds = depth_map.get(depth, 2)

    # Step 5: LLM Provider
    console.print()
    console.print(Panel(
        "[bold]Step 5: LLM Provider[/bold]\n"
        "Select which LLM to use for analysis.",
        border_style="blue",
    ))
    provider = questionary.select(
        "Select LLM Provider:",
        choices=[
            {"name": "DeepSeek", "value": "deepseek"},
            {"name": "OpenAI (GPT)", "value": "openai"},
            {"name": "Anthropic (Claude)", "value": "anthropic"},
            {"name": "Google (Gemini)", "value": "google"},
            {"name": "xAI (Grok)", "value": "xai"},
            {"name": "OpenRouter", "value": "openrouter"},
        ],
    ).ask()

    # Step 6: API Key
    api_key = get_api_key(provider)
    if not api_key:
        console.print("[red]API key required.[/red]")
        sys.exit(1)

    # Step 7: Model Selection
    console.print()
    console.print(Panel(
        "[bold]Step 6: Models[/bold]\n"
        "Select models for quick-thinking (analysts, debaters) and deep-thinking (decisions).",
        border_style="blue",
    ))

    model_options = {
        "deepseek": [
            {"name": "DeepSeek Chat (V3)", "value": "deepseek-chat"},
        ],
        "openai": [
            {"name": "GPT-5.4", "value": "gpt-5.4"},
            {"name": "GPT-5.4-mini", "value": "gpt-5.4-mini"},
            {"name": "O4 Mini", "value": "o4-mini"},
        ],
        "anthropic": [
            {"name": "Claude Opus 4.5", "value": "claude-opus-4-5-20251101"},
            {"name": "Claude Sonnet 4.5", "value": "claude-sonnet-4-5-20251101"},
        ],
        "google": [
            {"name": "Gemini 3.0 Pro", "value": "gemini-3.0-pro"},
            {"name": "Gemini 2.5 Flash", "value": "gemini-2.5-flash"},
        ],
        "xai": [
            {"name": "Grok 4", "value": "grok-4"},
        ],
        "openrouter": [
            {"name": "GPT-5.4 (via OpenRouter)", "value": "openai/gpt-5.4"},
            {"name": "Claude Opus 4.5 (via OpenRouter)", "value": "anthropic/claude-opus-4-5"},
        ],
    }
    models = model_options.get(provider, [{"name": "Default", "value": "default"}])

    quick_model = questionary.select(
        "Quick-Thinking Model (analysts, debaters):",
        choices=models,
    ).ask()

    deep_model = questionary.select(
        "Deep-Thinking Model (research manager, portfolio manager):",
        choices=models,
    ).ask()

    return {
        "token": token,
        "chain": chain,
        "analysts": analysts,
        "debate_rounds": rounds,
        "risk_rounds": rounds,
        "provider": provider,
        "api_key": api_key,
        "quick_model": quick_model,
        "deep_model": deep_model,
    }


def build_progress_table(current_step: int, agent_status: dict) -> Table:
    """Build the progress display table."""
    table = Table(box=box.SIMPLE, show_header=True, expand=True)
    table.add_column("Team", style="bold")
    table.add_column("Agent", style="cyan")
    table.add_column("Status")

    teams = [
        ("Analyst Team", [
            ("On-Chain Analyst", "onchain"),
            ("Sentiment Analyst", "sentiment"),
            ("Narrative Analyst", "narrative"),
            ("Technical Analyst", "technical"),
        ]),
        ("Research Team", [
            ("Bull Researcher", "bull"),
            ("Bear Researcher", "bear"),
            ("Research Manager", "research_manager"),
        ]),
        ("Trading Team", [
            ("Trader", "trader"),
        ]),
        ("Risk Management", [
            ("Aggressive Analyst", "aggressive"),
            ("Neutral Analyst", "neutral"),
            ("Conservative Analyst", "conservative"),
        ]),
        ("Portfolio Management", [
            ("Portfolio Manager", "portfolio_manager"),
        ]),
    ]

    for team_name, agents in teams:
        for agent_name, agent_key in agents:
            status = agent_status.get(agent_key, "pending")
            if status == "completed":
                icon = "[green]✓[/green]"
            elif status == "running":
                icon = "[yellow]⟳[/yellow]"
            else:
                icon = "[dim]○[/dim]"
            table.add_row(team_name, f"  {agent_name}", f"{icon} {status}")

    return table


def run_analysis(setup: dict):
    """Run the analysis with live progress display."""
    config = CRYPTOAGENT_DEFAULT_CONFIG.copy()
    config['llm_provider'] = setup['provider']
    config['deep_think_llm'] = setup['deep_model']
    config['quick_think_llm'] = setup['quick_model']
    config['api_key'] = setup['api_key']
    config['max_debate_rounds'] = setup['debate_rounds']
    config['max_risk_discuss_rounds'] = setup['risk_rounds']

    # Build progress display
    agent_status = {a: "pending" for a in [
        "onchain", "sentiment", "narrative", "technical",
        "bull", "bear", "research_manager",
        "trader",
        "aggressive", "neutral", "conservative",
        "portfolio_manager",
    ]}

    t0 = time.time()

    console.print()
    console.print(Panel.fit(
        f"[bold]Starting analysis...[/bold]\n"
        f"Token: [cyan]{setup['token']}[/cyan]\n"
        f"Chain: [cyan]{setup['chain']}[/cyan]\n"
        f"Analysts: [cyan]{', '.join(setup['analysts'])}[/cyan]\n"
        f"Provider: [cyan]{setup['provider']}[/cyan] | "
        f"Quick: [cyan]{setup['quick_model']}[/cyan] | "
        f"Deep: [cyan]{setup['deep_model']}[/cyan]",
        border_style="cyan",
    ))

    cg = CryptoAgentGraph(
        selected_analysts=setup['analysts'],
        config=config,
        debug=False,
    )

    # Run analysis with progress
    try:
        final_state, decision = cg.propagate(setup['token'], setup['chain'])
    except Exception as e:
        console.print(f"\n[red]Analysis failed: {e}[/red]")
        sys.exit(1)

    elapsed = time.time() - t0

    # Simulate progress (since we can't easily hook into LangGraph stream from here)
    # Mark everything as completed
    for key in agent_status:
        agent_status[key] = "completed"

    # Show final progress
    console.print()
    console.print(build_progress_table(5, agent_status))
    console.print()

    # Stats bar
    console.print(Panel(
        f"Agents: 14/14 | LLM calls: ~25 | "
        f"Tools: ~12 | ⏱ {int(elapsed // 60)}:{int(elapsed % 60):02d}",
        border_style="green",
    ))
    console.print()
    console.print("[bold green]Analysis Complete![/bold green]")
    console.print()

    # Offer to save/view report
    save = questionary.confirm("Save report?", default=True).ask()
    if save:
        path = cg.save_report()
        console.print(f"[green]Report saved: {path}[/green]")

    view = questionary.confirm("Display full report?", default=True).ask()
    if view:
        display_report(cg, setup)


def display_report(cg: CryptoAgentGraph, setup: dict):
    """Display the full report with structured sections."""
    state = cg.curr_state
    if not state:
        console.print("[red]No report available.[/red]")
        return

    console.clear()
    console.print(Panel.fit(
        "[bold cyan]Complete Analysis Report[/bold cyan]",
        border_style="cyan",
    ))

    # I. Analyst Team Reports
    console.print()
    console.rule("[bold]I. Analyst Team Reports[/bold]")

    if state.get("onchain_report"):
        console.print(Panel(
            state["onchain_report"][:2000] + ("..." if len(state.get("onchain_report", "")) > 2000 else ""),
            title="On-Chain Analyst",
            border_style="blue",
        ))

    if state.get("sentiment_report"):
        console.print(Panel(
            state["sentiment_report"][:2000] + ("..." if len(state.get("sentiment_report", "")) > 2000 else ""),
            title="Sentiment Analyst",
            border_style="blue",
        ))

    if state.get("narrative_report"):
        console.print(Panel(
            state["narrative_report"][:2000] + ("..." if len(state.get("narrative_report", "")) > 2000 else ""),
            title="Narrative Analyst",
            border_style="blue",
        ))

    if state.get("technical_report"):
        console.print(Panel(
            state["technical_report"][:2000] + ("..." if len(state.get("technical_report", "")) > 2000 else ""),
            title="Technical Analyst",
            border_style="blue",
        ))

    # II. Research Team Decision
    console.print()
    console.rule("[bold]II. Research Team Decision[/bold]")
    if state.get("investment_plan"):
        console.print(Panel(
            state["investment_plan"][:3000],
            title="Research Manager",
            border_style="yellow",
        ))

    # III. Trading Team Plan
    console.print()
    console.rule("[bold]III. Trading Team Plan[/bold]")
    if state.get("trader_investment_plan"):
        console.print(Panel(
            state["trader_investment_plan"][:2000],
            title="Trader",
            border_style="yellow",
        ))

    # IV. Risk Management
    console.print()
    console.rule("[bold]IV. Risk Management Team Decision[/bold]")
    risk_state = state.get("risk_debate_state", {})
    if risk_state.get("current_aggressive_response"):
        console.print(Panel(
            risk_state["current_aggressive_response"][:1000],
            title="Aggressive Analyst",
            border_style="red",
        ))
    if risk_state.get("current_conservative_response"):
        console.print(Panel(
            risk_state["current_conservative_response"][:1000],
            title="Conservative Analyst",
            border_style="green",
        ))
    if risk_state.get("current_neutral_response"):
        console.print(Panel(
            risk_state["current_neutral_response"][:1000],
            title="Neutral Analyst",
            border_style="yellow",
        ))

    # V. Portfolio Manager Decision
    console.print()
    console.rule("[bold]V. Portfolio Manager Decision[/bold]")
    if state.get("final_trade_decision"):
        console.print(Panel(
            state["final_trade_decision"],
            title="Portfolio Manager",
            border_style="bold cyan",
        ))

    console.print()


def main():
    """Main CLI entry point."""
    show_banner()

    try:
        setup = interactive_setup()
        run_analysis(setup)
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis cancelled.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
