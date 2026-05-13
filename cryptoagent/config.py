"""CryptoAgent configuration — mirrors TradingAgents' config system, adapted for crypto."""

import os
from pathlib import Path

CRYPTOAGENT_DEFAULT_CONFIG = {
    # ── LLM ──────────────────────────────────────────────
    "llm_provider": os.getenv("CRYPTOAGENT_LLM_PROVIDER", "deepseek"),
    "deep_think_llm": os.getenv("CRYPTOAGENT_DEEP_THINK_LLM", "deepseek-chat"),
    "quick_think_llm": os.getenv("CRYPTOAGENT_QUICK_THINK_LLM", "deepseek-chat"),
    "backend_url": os.getenv("CRYPTOAGENT_BACKEND_URL", None),
    "max_debate_rounds": int(os.getenv("CRYPTOAGENT_MAX_DEBATE_ROUNDS", "2")),
    "max_risk_discuss_rounds": int(os.getenv("CRYPTOAGENT_MAX_RISK_DISCUSS_ROUNDS", "2")),
    "max_recur_limit": int(os.getenv("CRYPTOAGENT_MAX_RECUR_LIMIT", "100")),
    "output_language": os.getenv("CRYPTOAGENT_OUTPUT_LANGUAGE", "English"),

    # ── Crypto-Specific ──────────────────────────────────
    "chains": ["ethereum", "base", "arbitrum", "solana", "bsc"],
    "min_liquidity_usd": float(os.getenv("CRYPTOAGENT_MIN_LIQUIDITY_USD", "100000")),
    "min_volume_24h_usd": float(os.getenv("CRYPTOAGENT_MIN_VOLUME_24H_USD", "50000")),
    "max_holder_concentration": float(os.getenv("CRYPTOAGENT_MAX_HOLDER_CONCENTRATION", "0.60")),
    "min_token_age_hours": int(os.getenv("CRYPTOAGENT_MIN_TOKEN_AGE_HOURS", "1")),

    # ── Trading ──────────────────────────────────────────
    "max_position_size_usd": float(os.getenv("CRYPTOAGENT_MAX_POSITION_SIZE_USD", "1000")),
    "max_portfolio_risk_pct": float(os.getenv("CRYPTOAGENT_MAX_PORTFOLIO_RISK_PCT", "0.25")),
    "max_slippage_bps": int(os.getenv("CRYPTOAGENT_MAX_SLIPPAGE_BPS", "300")),
    "default_stop_loss_pct": float(os.getenv("CRYPTOAGENT_DEFAULT_STOP_LOSS_PCT", "0.15")),
    "default_take_profit_pct": float(os.getenv("CRYPTOAGENT_DEFAULT_TAKE_PROFIT_PCT", "1.0")),

    # ── AST Bridge ───────────────────────────────────────
    "ast_ipc_path": os.getenv("CRYPTOAGENT_AST_IPC_PATH", "/tmp/ast-execution.sock"),
    "ast_http_url": os.getenv("CRYPTOAGENT_AST_HTTP_URL", "http://localhost:9090"),
    "ast_execution_enabled": os.getenv("CRYPTOAGENT_AST_EXECUTION_ENABLED", "false").lower() == "true",

    # ── Data ─────────────────────────────────────────────
    "data_cache_dir": os.path.expanduser(
        os.getenv("CRYPTOAGENT_DATA_CACHE_DIR", "~/.cryptoagent/cache")
    ),
    "results_dir": os.path.expanduser(
        os.getenv("CRYPTOAGENT_RESULTS_DIR", "~/.cryptoagent/results")
    ),
    "memory_log_path": os.path.expanduser(
        os.getenv("CRYPTOAGENT_MEMORY_LOG_PATH", "~/.cryptoagent/memory/trading_memory.md")
    ),
    "dex_api_url": os.getenv("CRYPTOAGENT_DEX_API_URL", "https://api.dexscreener.com"),
    "goplus_api_url": os.getenv("CRYPTOAGENT_GOPLUS_API_URL", "https://api.gopluslabs.io/api/v1"),
    "coingecko_api_url": os.getenv("CRYPTOAGENT_COINGECKO_API_URL", "https://api.coingecko.com/api/v3"),
    "defillama_api_url": os.getenv("CRYPTOAGENT_DEFILLAMA_API_URL", "https://api.llama.fi"),

}


def get_config() -> dict:
    """Return the current config. Can be overridden per-session."""
    return CRYPTOAGENT_DEFAULT_CONFIG.copy()
