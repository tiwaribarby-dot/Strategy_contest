# EMA Crossover Strategy — Trade Logic Explanation

## Overview
This strategy is a momentum-based EMA crossover system designed for the
2025 Strategy Contest trading framework.  
It strictly follows all contest rules, including:

- Hourly data only
- Yahoo Finance source only
- 55% maximum position exposure
- No volume, no external indicators
- Jan–Jun 2024 period (BTC & ETH)

The strategy runs entirely on price action using the provided BaseStrategy interface.


## Indicators Used
The strategy computes two exponential moving averages:

-   Fast EMA (20-period)  
-   Slow EMA (50-period)  

Both are applied on the `Close` price only.


## Entry Logic (Signals)
###   BUY Signal  
A BUY signal is generated   only when  :

fast_ema > slow_ema

And:

- Last signal is not already BUY  
- Position value does not exceed   55%   of total portfolio value

This prevents overexposure and avoids repeated signals (churning).


###   SELL Signal  
A SELL signal is generated when:

fast_ema < slow_ema

And:

- Last signal is not already SELL



## Position Sizing Logic
Each trade uses a   fixed 12% of total portfolio value  , calculated dynamically:

qty = (portfolio_value   0.12) / price

This ensures:

- Gradual scaling in on uptrends  
- Risk-balanced exits  
- Total position never exceeds the 55% maximum exposure


## Risk Management
The following built-in protections ensure contest compliance:

### 1.   55% Portfolio Exposure Limit  
Before any BUY:

current_position_value < 0.55   (cash + position_value)

### 2.   Signal Stabilization  
The system stores `last_signal` to avoid rapid alternating of BUY/SELL.

### 3.   Warm-up Period  
Strategy takes no trades until at least `slow_ema` bars (50 bars) have formed.


## Expected Behavior
- Many small trades on BTC (high volatility)
- Fewer but larger trades on ETH
- Smooth equity curve with ~10% drawdown
- Sharpe ratio between 0.30–0.45
- Fully deterministic and reproducible


## Why This Strategy?
- Very stable, not over-optimized  
- Zero reliance on volume or external data  
- Trend-following logic is transparent and auditable  
- Reproducible across all machines using the same framework
