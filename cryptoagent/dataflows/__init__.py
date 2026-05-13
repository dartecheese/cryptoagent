"""Data layer configuration — bridges cryptoagent.config to dataflows."""

from cryptoagent.config import CRYPTOAGENT_DEFAULT_CONFIG

_config = CRYPTOAGENT_DEFAULT_CONFIG.copy()


def set_config(config: dict) -> None:
    """Update the dataflows config (called by TradingGraph at init)."""
    global _config
    _config.update(config)


def get_config() -> dict:
    """Get current dataflows config."""
    return _config
