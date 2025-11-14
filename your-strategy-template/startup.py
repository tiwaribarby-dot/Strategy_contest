#!/usr/bin/env python3
"""Startup script for EMA + Volatility Strategy
Version: 1.0
"""

import sys
import os
from datetime import datetime

# Locate base bot template
base_path = os.path.join(os.path.dirname(__file__), '..', 'base-bot-template')
if not os.path.exists(base_path):
    base_path = '/app/base'
sys.path.insert(0, base_path)

from universal_bot import UniversalBot
import your_strategy  # registers the strategy automatically


def main():
    # Load config path
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not config_path or not os.path.exists(config_path):
        print("⚠No config file provided. Using default ./config.json")
        config_path = os.path.join(os.path.dirname(__file__), "config.json")

    # Log startup info
    print(f"[{datetime.utcnow().isoformat()}] Starting Hybrid EMA + Volatility Strategy v1.0")

    # Initialize and run bot
    bot = UniversalBot(config_path)
    strategy_name = bot.config.get("strategy", "ema_volatility_strategy")
    print(f"Running {strategy_name} on {bot.config.symbol}")
    bot.run()


if __name__ == "__main__":
    main()