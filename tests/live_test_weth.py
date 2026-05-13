#!/usr/bin/env python3
"""Live test — all 4 analysts. Must succeed for commit gate."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging, time

t0 = time.time()
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

from cryptoagent.config import CRYPTOAGENT_DEFAULT_CONFIG
from cryptoagent.graph.trading_graph import CryptoAgentGraph

config = CRYPTOAGENT_DEFAULT_CONFIG.copy()
config['llm_provider'] = 'deepseek'
config['deep_think_llm'] = 'deepseek-chat'
config['quick_think_llm'] = 'deepseek-chat'
config['api_key'] = os.getenv('DEEPSEEK_API_KEY', '')
config['max_debate_rounds'] = 1
config['max_risk_discuss_rounds'] = 1

print(f"[{time.strftime('%H:%M:%S')}] Starting 4-analyst WETH analysis...", flush=True)

cg = CryptoAgentGraph(
    selected_analysts=['onchain', 'sentiment', 'narrative', 'technical'],
    config=config,
    debug=False,
)

final_state, decision = cg.propagate('0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2', 'ethereum')

elapsed = time.time() - t0
path = cg.save_report()

print(f"[{time.strftime('%H:%M:%S')}] DONE in {elapsed:.0f}s", flush=True)
print(f"Report: {path}", flush=True)
print(f"Decision: {decision[:300]}...", flush=True)
print("COMMIT_GATE: PASS", flush=True)
