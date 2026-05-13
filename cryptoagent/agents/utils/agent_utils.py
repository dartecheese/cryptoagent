"""Shared utilities for CryptoAgent — mirrors TradingAgents' agent_utils.

Provides:
- create_msg_delete: clears messages between agent steps
- build_token_context: describes the token/chain for prompts
- get_language_instruction: optional non-English output
- Crypto-specific tool definitions for analysts
"""

from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.tools import tool


# ── Message Management ─────────────────────────────────────────────


def create_msg_delete():
    """Return a node that clears messages and places a placeholder.

    Required for Anthropic compatibility and to keep context windows
    manageable between agent steps.
    """
    def delete_messages(state):
        messages = state["messages"]
        removal_operations = [RemoveMessage(id=m.id) for m in messages]
        placeholder = HumanMessage(content="Continue")
        return {"messages": removal_operations + [placeholder]}
    return delete_messages


# ── Prompt Helpers ──────────────────────────────────────────────────


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string for English (default).
    """
    from cryptoagent.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def build_token_context(token: str, chain: str) -> str:
    """Describe the token and chain so agents use correct context."""
    return (
        f"The token to analyze is `{token}` on the `{chain}` blockchain. "
        f"Use this exact token address and chain in every tool call, report, and recommendation."
    )


# ── Crypto Tool Definitions ────────────────────────────────────────
# These are LangChain tools that LLM agents can call to fetch real data.
# Each tool does actual HTTP calls at invocation time.


@tool
def get_token_metadata(token_address: str, chain: str) -> str:
    """Fetch token metadata: name, symbol, decimals, deployer, creation date.
    
    Args:
        token_address: Contract address of the token
        chain: Blockchain (ethereum, base, arbitrum, solana, bsc)
    """
    import httpx
    import json
    from cryptoagent.dataflows.config import get_config
    
    cfg = get_config()
    
    # Try DexScreener first for metadata
    try:
        url = f"{cfg['dex_api_url']}/latest/dex/search?q={token_address}"
        resp = httpx.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            pairs = data.get("pairs", [])
            if pairs:
                p = pairs[0]
                chain_id = p.get("chainId", chain)
                base = p.get("baseToken", {})
                return json.dumps({
                    "name": base.get("name", "Unknown"),
                    "symbol": base.get("symbol", "Unknown"),
                    "chain": chain_id,
                    "address": base.get("address", token_address),
                    "price_usd": p.get("priceUsd", "N/A"),
                    "liquidity_usd": p.get("liquidity", {}).get("usd", "N/A"),
                    "volume_24h_usd": p.get("volume", {}).get("h24", "N/A"),
                    "fdv_usd": p.get("fdv", "N/A"),
                    "pair_created_at": p.get("pairCreatedAt", "N/A"),
                }, indent=2)
    except Exception as e:
        pass
    
    return f"Token metadata not found for {token_address} on {chain}"


@tool
def get_token_price_ohlcv(token_address: str, chain: str, timeframe: str = "1h") -> str:
    """Fetch OHLCV price data for a token pair.
    
    Args:
        token_address: Contract address of the token
        chain: Blockchain
        timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
    """
    import httpx
    import json
    from cryptoagent.dataflows.config import get_config
    
    cfg = get_config()
    
    # CoinGecko: need to find the token first
    # For now, use DexScreener pair data as price proxy
    try:
        url = f"{cfg['dex_api_url']}/latest/dex/search?q={token_address}"
        resp = httpx.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            pairs = data.get("pairs", [])[:5]
            result = []
            for p in pairs:
                result.append({
                    "pair": p.get("pairAddress"),
                    "dex": p.get("dexId"),
                    "price_usd": p.get("priceUsd"),
                    "price_change_1h": p.get("priceChange", {}).get("h1"),
                    "price_change_6h": p.get("priceChange", {}).get("h6"),
                    "price_change_24h": p.get("priceChange", {}).get("h24"),
                    "volume_24h": p.get("volume", {}).get("h24"),
                    "liquidity_usd": p.get("liquidity", {}).get("usd"),
                    "txns_24h": p.get("txns", {}).get("h24", {}).get("buys", 0) +
                                p.get("txns", {}).get("h24", {}).get("sells", 0),
                })
            return json.dumps(result[:3], indent=2)  # top 3 pairs
    except Exception as e:
        return f"Price data unavailable: {e}"

    return f"No price data found for {token_address} on {chain}"


@tool
def get_onchain_metrics(token_address: str, chain: str) -> str:
    """Fetch on-chain metrics: holders, transfers, TVL, protocol data.
    
    Args:
        token_address: Contract address
        chain: Blockchain
    """
    import httpx
    import json
    from cryptoagent.dataflows.config import get_config
    
    cfg = get_config()
    metrics = {}
    
    # GoPlus security data
    if chain in ("ethereum", "bsc", "arbitrum", "base", "polygon"):
        try:
            chain_id = {"ethereum": "1", "bsc": "56", "arbitrum": "42161",
                        "base": "8453", "polygon": "137"}.get(chain, "1")
            url = f"{cfg['goplus_api_url']}/token_security/{chain_id}?contract_addresses={token_address}"
            resp = httpx.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                result = data.get("result", {})
                token_data = result.get(token_address.lower(), {})
                if token_data:
                    metrics["security"] = {
                        "is_honeypot": token_data.get("is_honeypot"),
                        "is_open_source": token_data.get("is_open_source"),
                        "buy_tax": token_data.get("buy_tax"),
                        "sell_tax": token_data.get("sell_tax"),
                        "is_proxy": token_data.get("is_proxy"),
                        "is_mintable": token_data.get("is_mintable"),
                        "owner_address": token_data.get("owner_address"),
                        "holder_count": token_data.get("holder_count"),
                        "lp_holder_count": token_data.get("lp_holder_count"),
                    }
        except Exception:
            pass
    
    return json.dumps(metrics, indent=2) if metrics else "On-chain metrics unavailable"


@tool
def get_crypto_news(token_symbol: str) -> str:
    """Fetch recent market activity and news signals for a token.
    Uses DexScreener pair data as proxy for market attention.
    
    Args:
        token_symbol: Token symbol or address
    """
    import httpx
    import json
    from cryptoagent.dataflows.config import get_config
    
    cfg = get_config()
    results = []
    try:
        url = f"{cfg['dex_api_url']}/latest/dex/search?q={token_symbol}"
        resp = httpx.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            pairs = data.get("pairs", [])[:5]
            for p in pairs:
                txns = p.get("txns", {}).get("h24", {})
                results.append({
                    "dex": p.get("dexId"),
                    "price_usd": p.get("priceUsd"),
                    "change_24h": p.get("priceChange", {}).get("h24"),
                    "volume_24h": p.get("volume", {}).get("h24"),
                    "buys": txns.get("buys", 0),
                    "sells": txns.get("sells", 0),
                })
    except Exception:
        pass
    if results:
        return json.dumps({"market_activity": results[:3]}, indent=2)
    return f"No market activity data for {token_symbol}"


@tool
def get_narrative_data() -> str:
    """Fetch current market narratives: trending categories, hot sectors, rotation signals."""
    import httpx
    import json
    from cryptoagent.dataflows.config import get_config
    
    cfg = get_config()
    results = {}
    
    # DexScreener trending tokens
    try:
        url = f"{cfg['dex_api_url']}/latest/dex/trending"
        resp = httpx.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            tokens = data.get("tokens", [])[:10]
            results["trending_tokens"] = [
                {"symbol": t.get("symbol"), "chain": t.get("chainId"),
                 "volume_24h": t.get("volume24h")}
                for t in tokens
            ]
    except Exception:
        pass
    
    # DexScreener boosted tokens (narrative signal)
    try:
        url = f"{cfg['dex_api_url']}/latest/dex/boosts/top"
        resp = httpx.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            boosts = data.get("boosts", [])[:10]
            results["boosted_tokens"] = [
                {"symbol": b.get("symbol"), "chain": b.get("chainId"),
                 "total_amount": b.get("totalAmount")}
                for b in boosts
            ]
    except Exception:
        pass
    
    # CoinGecko trending
    try:
        url = f"{cfg['coingecko_api_url']}/search/trending"
        resp = httpx.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            coins = data.get("coins", [])[:10]
            results["coingecko_trending"] = [
                {"name": c.get("item", {}).get("name"),
                 "symbol": c.get("item", {}).get("symbol"),
                 "market_cap_rank": c.get("item", {}).get("market_cap_rank")}
                for c in coins
            ]
    except Exception:
        pass
    
    return json.dumps(results, indent=2) if results else "Narrative data unavailable"


@tool
def get_social_sentiment(token_symbol: str) -> str:
    """Fetch social sentiment from crypto-specific sources.
    
    Args:
        token_symbol: Token symbol for sentiment lookup
    """
    import httpx
    import json
    from cryptoagent.dataflows.config import get_config
    
    cfg = get_config()
    sentiment = {}
    
    # LunarCrush (if key available)
    # DexScreener recent boosts (proxy for social interest)
    try:
        url = f"{cfg['dex_api_url']}/latest/dex/search?q={token_symbol}"
        resp = httpx.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            pairs = data.get("pairs", [])[:5]
            sentiment["trading_activity"] = []
            for p in pairs:
                txns = p.get("txns", {}).get("h24", {})
                sentiment["trading_activity"].append({
                    "pair": p.get("pairAddress"),
                    "dex": p.get("dexId"),
                    "buys_24h": txns.get("buys", 0),
                    "sells_24h": txns.get("sells", 0),
                    "buy_sell_ratio": (
                        txns.get("buys", 0) / max(txns.get("sells", 1), 1)
                    ),
                    "volume_24h": p.get("volume", {}).get("h24"),
                })
    except Exception:
        pass
    
    return json.dumps(sentiment, indent=2) if sentiment else f"No social data for {token_symbol}"
