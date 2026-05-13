"""Trading memory log — captures decisions and outcomes for learning.

Crypto-adapted from TradingAgents' memory.py. Tracks:
- Token, chain, analysis timestamp
- Rating, conviction, thesis
- Outcome (when known)
- Lessons learned
"""

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


class TradingMemoryLog:
    """Persistent trading memory — what worked, what didn't, what we learned."""

    def __init__(self, config: dict):
        self.memory_path = Path(config.get("memory_log_path",
                              "~/.cryptoagent/memory/trading_memory.md")).expanduser()
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

    def log_decision(
        self,
        token: str,
        chain: str,
        rating: str,
        conviction: str,
        thesis: str,
        price_target: Optional[float] = None,
        time_horizon: Optional[str] = None,
    ) -> None:
        """Record a trading decision to the memory log.

        Called after the Portfolio Manager makes a final decision.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        entry = f"""### {timestamp} — {token} ({chain})

- **Rating**: {rating}
- **Conviction**: {conviction}
- **Time Horizon**: {time_horizon or 'N/A'}
- **Price Target**: {f'${price_target:.6f}' if price_target else 'N/A'}

**Thesis**: {thesis}

---
"""
        with open(self.memory_path, "a") as f:
            f.write(entry)

    def get_recent_context(self, token: str | None = None, limit: int = 5) -> str:
        """Get recent trading memory for context injection.

        If token is provided, returns decisions for that specific token.
        Otherwise returns the most recent decisions across all tokens.
        """
        if not self.memory_path.exists():
            return ""

        with open(self.memory_path) as f:
            content = f.read()

        entries = content.split("---")
        if token:
            entries = [e for e in entries if token in e]
        entries = entries[-limit:]

        return "\n---\n".join(entries).strip()

    def log_outcome(
        self,
        token: str,
        chain: str,
        rating: str,
        outcome: str,  # "Profit", "Loss", "Breakeven", "Not Executed"
        pnl_pct: Optional[float] = None,
        notes: str = "",
    ) -> None:
        """Record the outcome of a past decision for learning."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        entry = f"""### OUTCOME: {timestamp} — {token} ({chain})

- **Original Rating**: {rating}
- **Outcome**: {outcome}
- **PnL**: {f'{pnl_pct:+.1f}%' if pnl_pct is not None else 'N/A'}

**Reflection**: {notes}

---
"""
        with open(self.memory_path, "a") as f:
            f.write(entry)
