#!/usr/bin/env python3
"""
Backtest runner for ena + Volatility Strategy.

Features:
- Simulated execution delay (n bars), slippage (pct), and fees (pct per side).
- Proper use of Signal.size: interpreted as fraction of available portfolio (0..1)
  or absolute quantity if >1 (will be treated as shares/coins).
- Computes equity curve, PnL, Sharpe (annualized), max drawdown, trade summary.
- Prints results and writes a simple markdown backtest report (backtest_report.md).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os
import math
import json
from typing import List, Dict, Any, Optional
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'your-strategy-template'))

# adjust import path to find base template & your strategy
import sys
base_path = os.path.join(os.path.dirname(__file__), '..', 'base-bot-template')
if not os.path.exists(base_path):
    base_path = '/app/base'
sys.path.insert(0, base_path)

# Import the strategy class (ensure your_strategy.py registers the class)
from your_strategy import EmaVolatilityStrategy
from exchange_interface import MarketSnapshot  # optional, we create local snapshots

# ---------------------------
# Configurable backtest params
# ---------------------------
DEFAULT_CONFIG = {
    "short_window": 20,
    "long_window": 100,
    "vol_window": 30,
    "vol_threshold": 0.03,
    "max_position": 0.8,            # max fraction of portfolio deployed per trade
    "cooldown_minutes": 60,
    # Execution simulation
    "execution_delay_bars": 1,      # trade fills 1 bar after signal
    "slippage_pct": 0.0005,         # 0.05% slippage relative to price
    "fee_pct": 0.00075,             # 0.075% fee each side
    # Backtest
    "starting_cash": 10_000.0,
    "min_trades_required": 10,
    # Timeframe assumed for annualization (daily series => 252 trading days)
    "annualization_factor": 252,
}


# ---------------------------
# Helper utilities
# ---------------------------
def apply_slippage_and_fee(price: float, side: str, slippage_pct: float, fee_pct: float) -> float:
    """Return effective execution price after slippage and include fees externally."""
    if side.lower() == "buy":
        exec_price = price * (1 + slippage_pct)
    else:
        exec_price = price * (1 - slippage_pct)
    # fee will be applied on trade value separately
    return exec_price


def compute_sharpe(returns: np.ndarray, annual_factor: float) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=0)
    # annualize
    sr = (mean_ret * annual_factor) / (std_ret * math.sqrt(annual_factor))
    return float(sr)


def max_drawdown(equity: np.ndarray) -> float:
    highwater = np.maximum.accumulate(equity)
    drawdowns = (equity - highwater) / highwater
    return float(np.min(drawdowns))  # negative number


# ---------------------------
# Backtest engine
# ---------------------------
class BacktestEngine:
    def __init__(self, df: pd.DataFrame, strategy_config: Dict[str, Any], sim_config: Dict[str, Any]):
        self.df = df.reset_index(drop=True)
        self.strategy_config = strategy_config
        self.sim_config = sim_config
        self.exchange = None  # not used, strategy receives exchange arg in constructor but we don't call it
        self.strategy = EmaVolatilityStrategy(self.strategy_config, exchange=None)

        # portfolio state
        self.cash = float(sim_config.get("starting_cash", DEFAULT_CONFIG["starting_cash"]))
        self.position = 0.0        # quantity of asset (coins)
        self.position_entry_price = None
        self.equity_curve: List[float] = []
        self.trades: List[Dict[str, Any]] = []

        # For execution delay
        self.pending_order: Optional[Dict[str, Any]] = None  # {side, size, submit_index}

    def _make_snapshot(self, index: int, lookback: int = 200) -> MarketSnapshot:
        """
        Create a MarketSnapshot-like object expected by strategy.generate_signal.
        We provide `prices` as a simple python list and `current_price`.
        """
        prices = self.df['Close'].iloc[max(0, index - lookback + 1): index + 1].tolist()
        current_price = float(self.df['Close'].iloc[index])
        ts = pd.to_datetime(self.df['Date'].iloc[index]) if 'Date' in self.df.columns else datetime.now(timezone.utc)
        return MarketSnapshot(symbol=self.df.get('Symbol', pd.Series(["P"]*len(self.df))).iloc[index],
                              prices=prices,
                              current_price=current_price,
                              timestamp=ts)

    def _portfolio_position_size_fraction(self) -> float:
        """Return current position size as fraction of portfolio value."""
        current_price = float(self.df['Close'].iloc[self.current_index])
        total_value = self.cash + self.position * current_price
        if total_value <= 0:
            return 0.0
        return (self.position * current_price) / total_value

    def _execute_pending_if_due(self, index: int):
        """If there's a pending order and delay elapsed, execute it at this bar's price."""
        if not self.pending_order:
            return
        submit_index = self.pending_order['submit_index']
        delay = self.sim_config.get("execution_delay_bars", 1)
        if index - submit_index >= delay:
            # Execute now
            side = self.pending_order['side']
            size = self.pending_order['size']
            price = float(self.df['Close'].iloc[index])
            slippage = self.sim_config.get("slippage_pct", 0.0)
            fee_pct = self.sim_config.get("fee_pct", 0.0)
            exec_price = apply_slippage_and_fee(price, side, slippage, fee_pct)

            if side == "buy":
                # interpret size: if 0 < size <=1 -> fraction of portfolio; if >1 -> absolute quantity
                if 0 < size <= 1:
                    # allocate fraction of total equity
                    total_equity = self.cash + self.position * price
                    alloc = total_equity * size
                    quantity = (alloc * (1 - fee_pct)) / exec_price
                else:
                    quantity = float(size)
                cost = quantity * exec_price
                fee = cost * fee_pct
                total_cost = cost + fee
                if total_cost > self.cash + 1e-9:
                    # not enough cash: buy as much as possible
                    quantity = (self.cash) / (exec_price * (1 + fee_pct))
                    cost = quantity * exec_price
                    fee = cost * fee_pct
                    total_cost = cost + fee
                self.position += quantity
                self.cash -= total_cost
                self.position_entry_price = exec_price
            elif side == "sell":
                if 0 < size <= 1:
                    # fraction of current position to sell
                    quantity = self.position * size
                else:
                    quantity = min(self.position, float(size))
                proceeds = quantity * exec_price
                fee = proceeds * fee_pct
                net = proceeds - fee
                self.position -= quantity
                self.cash += net
                if self.position <= 0:
                    self.position_entry_price = None
            # record trade
            self.trades.append({
                "index": index,
                "timestamp": str(self.df['Date'].iloc[index]) if 'Date' in self.df.columns else str(datetime.now(timezone.utc)),
                "side": side,
                "price": exec_price,
                "quantity": quantity,
                "cash_after": self.cash,
                "position_after": self.position,
            })
            # clear pending
            self.pending_order = None

    def run(self):
        n = len(self.df)
        self.current_index = 0
        for i in range(n):
            self.current_index = i
            # 1) If pending order is due, execute it first using this bar's price
            self._execute_pending_if_due(i)

            # 2) Build a MarketSnapshot and portfolio object for strategy
            snapshot = self._make_snapshot(i)
            class PortfolioObj:
                pass
            portfolio_obj = PortfolioObj()
            portfolio_obj.cash = self.cash
            portfolio_obj.quantity = self.position
            portfolio_obj.position_size = self._portfolio_position_size_fraction()

            # 3) Ask strategy for signal
            try:
                signal = self.strategy.generate_signal(snapshot, portfolio_obj)
            except Exception as exc:
                print(f"Strategy raised exception at index {i}: {exc}")
                signal = None

            # 4) If strategy returned buy/sell, enqueue a pending order (simulate delay)
            if signal and signal.action in ("buy", "sell"):
                size = getattr(signal, "size", self.strategy_config.get("max_position", 0.5))
                # create pending order
                self.pending_order = {
                    "side": signal.action,
                    "size": float(size),
                    "submit_index": i
                }
            # 5) record equity for this bar
            current_price = float(self.df['Close'].iloc[i])
            total_value = self.cash + self.position * current_price
            self.equity_curve.append(total_value)

        # finalize: ensure any renaining pending order executes at last bar price
        if self.pending_order:
            self._execute_pending_if_due(n - 1)

        # metrics
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        final_value = equity[-1] if len(equity) else self.cash
        total_pnl = final_value - self.sim_config.get("starting_cash", DEFAULT_CONFIG["starting_cash"])
        sharpe = compute_sharpe(returns, self.sim_config.get("annualization_factor", DEFAULT_CONFIG["annualization_factor"]))
        mdd = max_drawdown(equity)
        trade_count = len(self.trades)

        results = {
            "final_value": float(final_value),
            "total_pnl": float(total_pnl),
            "trade_count": int(trade_count),
            "sharpe": float(sharpe),
            "max_drawdown": float(mdd),
        }
        return results, equity, returns, self.trades


# ---------------------------
# Runner entrypoint
# ---------------------------
# ---------------------------
# Runner entrypoint
# ---------------------------
def run_backtest(csv_path: str, out_report: str = "reports/backtest_report.md", config_path: Optional[str] = None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Correct for your CSV (no header skipping)
    df = pd.read_csv(csv_path)

    # Ensure Close column exists
    if 'Close' not in df.columns:
        raise ValueError(f"CSV must contain a 'Close' column, found columns: {df.columns.tolist()}")

    # Rename Datetime -> Date (optional, only if your code uses 'Date')
    if 'Datetime' in df.columns and 'Date' not in df.columns:
        df = df.rename(columns={'Datetime': 'Date'})

    strategy_cfg = DEFAULT_CONFIG.copy()
    sim_cfg = DEFAULT_CONFIG.copy()

    #  Optional config load
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as fh:
            user_cfg = json.load(fh)
            strategy_cfg.update(user_cfg.get("strategy", {}))
            sim_cfg.update(user_cfg.get("sim", {}))

    # Run backtest
    engine = BacktestEngine(df, strategy_cfg, sim_cfg)
    results, equity, returns, trades = engine.run()

    # Print summary
    print("=== BACKTEST SUMMARY ===")
    print(f"Start cash: ${sim_cfg['starting_cash']:,}")
    print(f"Final value: ${results['final_value']:,.2f}")
    print(f"Total PnL: ${results['total_pnl']:,.2f} ({results['total_pnl'] / sim_cfg['starting_cash'] * 100:.2f}%)")
    print(f"Trades executed: {results['trade_count']}")
    print(f"Sharpe (ann): {results['sharpe']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown'] * 100:.2f}%")
    if results['trade_count'] < sim_cfg.get("min_trades_required", 10):
        print("⚠WARNING: trade count < required minimum. Consider tuning strategy parameters (vol_threshold, cooldown, max_position).")

    # Write markdown report
    os.makedirs(os.path.dirname(out_report), exist_ok=True)
    with open(out_report, 'w') as fh:
        fh.write("# Backtest Report\n\n")
        fh.write(f"**CSV:** {csv_path}\n\n")
        fh.write(f"- Start cash: ${sim_cfg['starting_cash']:,}\n")
        fh.write(f"- Final value: ${results['final_value']:,.2f}\n")
        fh.write(f"- Total PnL: ${results['total_pnl']:,.2f}\n")
        fh.write(f"- Trades executed: {results['trade_count']}\n")
        fh.write(f"- Sharpe (ann): {results['sharpe']:.2f}\n")
        fh.write(f"- Max Drawdown: {results['max_drawdown'] * 100:.2f}%\n")
        fh.write("\n## Trades\n\n")
        fh.write("| idx | timestamp | side | price | qty | cash_after | position_after |\n")
        fh.write("|---:|---|---|---:|---:|---:|---:|\n")
        for t in trades:
            fh.write(f"| {t['index']} | {t['timestamp']} | {t['side']} | {t['price']:.6f} | {t['quantity']:.6f} | {t['cash_after']:.2f} | {t['position_after']:.6f} |\n")

    print(f"Report written to {out_report}")
    return results, equity, returns, trades


# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run backtest for EmaVolatilityStrategy")
    parser.add_argument("csv", help="CSV file with Close column (ordered oldest->newest)")
    parser.add_argument("--config", help="Optional JSON config to override defaults", default=None)
    parser.add_argument("--out", help="Markdown report output path", default="reports/backtest_report.md")
    args = parser.parse_args()
    run_backtest(args.csv, out_report=args.out, config_path=args.config)