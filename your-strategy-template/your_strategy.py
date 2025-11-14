#!/usr/bin/env python3
"""
Simplified + Fixed EMA Strategy
- High trade frequency (10–25 trades)
- No cooldown (contest prefers more trades)
- 100% price-only logic (no extra data)
- Clean compatibility with MarketSnapshot
"""

from __future__ import annotations
import sys, os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

# -------------------------------------------------------
# Allow importing from base-bot-template (without renaming)
# -------------------------------------------------------
base_path = os.path.join(os.path.dirname(__file__), "..", "base-bot-template")
if os.path.exists(base_path) and base_path not in sys.path:
    sys.path.insert(0, base_path)

from strategy_interface import BaseStrategy, Signal, register_strategy
from exchange_interface import Exchange, MarketSnapshot


# =======================================================
#   FINAL CONTEST STRATEGY (HIGH-FREQUENCY VERSION)
# =======================================================
class EmaVolatilityStrategy(BaseStrategy):
    def __init__(self, config: Optional[Dict[str, Any]] = None, exchange: Optional[Exchange] = None):
        if config is None:
            config = {}

        super().__init__(config=config, exchange=exchange)

        self.short_window = int(config.get("short_window", 20))
        self.long_window  = int(config.get("long_window", 40))
        self.vol_window   = int(config.get("vol_window", 20))
        self.vol_threshold = float(config.get("vol_threshold", 0.03))

        self.max_exposure = float(config.get("max_exposure", 0.55))
        self.min_usd_trade = float(config.get("min_usd_trade", 20))

    # ---- helpers ----
    def _get_prices(self, market: MarketSnapshot) -> np.ndarray:
        """
        SAFE price extraction from MarketSnapshot
        Works for your runner + backtest CSV loader both.
        """
        if hasattr(market, "prices") and market.prices:
            return np.array(market.prices, dtype=float)
        if hasattr(market, "history") and market.history:
            return np.array(market.history, dtype=float)
        if hasattr(market, "df"):
            return market.df["close"].astype(float).values

        raise ValueError("MarketSnapshot contains no price series")

    # ---- strategy ----
    def generate_signal(self, market: MarketSnapshot, portfolio) -> Signal:
        prices = self._get_prices(market)

        if len(prices) < max(self.short_window, self.long_window, self.vol_window) + 2:
            return Signal("hold")

        s = pd.Series(prices)

        short_ema = s.ewm(span=self.short_window, adjust=False).mean().iloc[-1]
        long_ema  = s.ewm(span=self.long_window, adjust=False).mean().iloc[-1]

        volatility = s.pct_change().rolling(self.vol_window).std().iloc[-1]
        if pd.isna(volatility):
            volatility = 0.0

        current_price = float(market.current_price)
        cash = float(getattr(portfolio, "cash", 0.0) or 0.0)
        qty  = float(getattr(portfolio, "quantity", 0.0) or 0.0)

        # exposure limit
        portfolio_value = cash + qty * current_price
        max_allowed_exposure = portfolio_value * self.max_exposure
        current_exposure = qty * current_price
        usd_to_use = max_allowed_exposure - current_exposure

        if usd_to_use < 0:
            usd_to_use = 0
        usd_to_use = min(usd_to_use, cash)

        # ENTRY → More frequent because no cooldown
        if short_ema > long_ema and volatility < self.vol_threshold and usd_to_use >= self.min_usd_trade:
            size = usd_to_use / current_price
            return Signal("buy", size=size, reason="ema_up")

        # EXIT more frequent
        if short_ema < long_ema and qty > 0:
            return Signal("sell", size=qty, reason="ema_down")

        return Signal("hold")


# ======================================================
# Register strategy correctly
# ======================================================
register_strategy(
    "ema_volatility_strategy",
    factory=lambda config, exchange: EmaVolatilityStrategy(config=config, exchange=exchange),
                       )
