#!/usr/bin/env python3
"""CLI entry point for CryptoAgent.

Usage:
    cryptoagent TOKEN_ADDRESS --chain ethereum
    cryptoagent TOKEN_ADDRESS --chain solana --analysts onchain,technical,sentiment
    cryptoagent TOKEN_ADDRESS --report output.md
"""

import argparse
import logging
import sys
from pathlib import Path

from cryptoagent.config import CRYPTOAGENT_DEFAULT_CONFIG
from cryptoagent.graph.trading_graph import CryptoAgentGraph


def main():
    parser = argparse.ArgumentParser(
        description="CryptoAgent — Multi-Agent LLM Crypto Trading Analysis"
    )
    parser.add_argument("token", help="Token contract address or symbol to analyze")
    parser.add_argument(
        "--chain", "-c", default="ethereum",
        choices=["ethereum", "base", "arbitrum", "solana", "bsc"],
        help="Blockchain context (default: ethereum)"
    )
    parser.add_argument(
        "--analysts", "-a", default="onchain,sentiment,narrative,technical",
        help="Comma-separated analysts to include (default: all four)"
    )
    parser.add_argument(
        "--report", "-r", default=None,
        help="Save report to file (default: auto-generated in results dir)"
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--provider", default=None,
        help="LLM provider override"
    )
    parser.add_argument(
        "--deep-model", default=None,
        help="Deep thinking model override"
    )
    parser.add_argument(
        "--quick-model", default=None,
        help="Quick thinking model override"
    )
    parser.add_argument(
        "--debate-rounds", type=int, default=2,
        help="Max debate rounds (default: 2)"
    )
    parser.add_argument(
        "--risk-rounds", type=int, default=2,
        help="Max risk debate rounds (default: 2)"
    )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("cryptoagent").setLevel(logging.INFO)

    # Build config
    config = CRYPTOAGENT_DEFAULT_CONFIG.copy()
    if args.provider:
        config["llm_provider"] = args.provider
    if args.deep_model:
        config["deep_think_llm"] = args.deep_model
    if args.quick_model:
        config["quick_think_llm"] = args.quick_model
    config["max_debate_rounds"] = args.debate_rounds
    config["max_risk_discuss_rounds"] = args.risk_rounds

    # Parse analysts
    selected_analysts = [a.strip() for a in args.analysts.split(",")]
    valid = {"onchain", "sentiment", "narrative", "technical"}
    selected_analysts = [a for a in selected_analysts if a in valid]
    if not selected_analysts:
        print("Error: No valid analysts selected.", file=sys.stderr)
        sys.exit(1)

    print(f"🔍 CryptoAgent analyzing `{args.token}` on {args.chain}")
    print(f"   Analysts: {', '.join(selected_analysts)}")
    print(f"   Model: {config['llm_provider']} (deep: {config['deep_think_llm']}, quick: {config['quick_think_llm']})")
    print()

    # Run analysis
    cg = CryptoAgentGraph(
        selected_analysts=selected_analysts,
        config=config,
        debug=args.debug,
    )

    try:
        final_state, decision = cg.propagate(args.token, args.chain)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}", file=sys.stderr)
        if args.debug:
            raise
        sys.exit(1)

    # Print report
    print(cg.get_report())

    # Save report
    report_path = cg.save_report(args.report)
    print(f"\n📄 Report saved: {report_path}")

    # Print decision summary
    if decision:
        print(f"\n{'='*60}")
        print(f"FINAL DECISION: {decision[:500]}")
        print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
