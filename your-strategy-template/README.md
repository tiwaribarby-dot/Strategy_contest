# EMA Crossover Strategy
This repository contains my submission for the Strategy Contest.
The strategy is designed according to all mandatory contest rules, including
55% exposure limits, hourly data interval, and Yahoo Finance data source.

## Folder Structure

your-strategy-template/
│── your_strategy.py          ← Main strategy logic  
│── startup.py                ← Bot entry point  
│── Dockerfile                ← Minimal container for evaluation  
│── requirements.txt          ← Required dependencies  
│── README.md                 ← Documentation  

base-bot-template/            ← Framework provided by contest  
reports/                      ← Backtest results for BTC & ETH  
trade_logic_explanation.md    ← Full explanation of trading logic  

---

## How to Run Locally

### 1. Install dependencies

pip install -r requirements.txt

### 2. Run backtest

python reports/backtest_runner.py

### 3. Run bot

python startup.py


## Backtest Results

The strategy was backtested on:

- BTC-USD (Jan–Jun 2024, 1h)
- ETH-USD (Jan–Jun 2024, 1h)

Results include:

- Final portfolio value
- Sharpe ratio
- Max drawdown
- Trade count  
(Full details in `/reports/backtest_report_BTC.md` and `backtest_report_ETH.md`)


##  Contest Compliance

This strategy satisfies:

- ✓ 55% max exposure limit  
- ✓ At least 10 trades  
- ✓ Yahoo Finance hourly price data  
- ✓ No volume / external data  
- ✓ Fully reproducible backtests  
- ✓ Docker-ready execution  

