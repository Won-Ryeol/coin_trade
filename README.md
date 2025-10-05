# Coin Trade Backtester

## Overview
Coin Trade is a Bitcoin-focused backtesting harness that emulates realistic order handling, configurable risk controls, and interactive reporting. It samples 15m OHLCV datasets, applies squeeze-and-trend filters, and produces metrics, trade logs, and Plotly dashboards for rapid strategy iteration.

## Key Features
- Bollinger-Keltner squeeze with optional RSI mean-reversion, higher-timeframe trend, and daily loss caps.
- ATR-driven take-profit/stop-loss sizing with fee and slippage buffers, plus auto-throttle to target trade counts.
- Deterministic CLI for research, production, and sweep workloads; supports JSON/YAML overrides and on-the-fly parameter tweaks.
- Equity curves, trade exports, and Plotly HTML visualizations saved per run alongside computed metrics.

## Repository Layout
- `coin_trade/` - core modules (`signals.py`, `trades.py`, `metrics.py`, `plotting.py`, helpers).
- `data/` - input CSVs (15m OHLCV); add regression fixtures under `data/samples/`.
- `artifacts/` - generated outputs (`trades.csv`, `equity.csv`, `metrics.json`, `backtest.html`).
- `tests/` - pytest suite covering loaders, signal gates, trade construction, and cooldown logic.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy plotly pytest pyyaml
```
Place your 15m CSV files in `data/` before running the CLI.

## Running Backtests
```bash
python run_backtest.py --mode research --sample-k 3 --random-seed 42 --min-rows 60
python run_backtest.py --sweep sweep.yaml --sample-k 2 --random-seed 7
```
Each command prints selected files, writes artifacts to `artifacts/`, and skips metrics if the minimum trade guard is not met. Use `--set key=value` or `--throttle-set key=value` to override strategy or auto-throttle parameters.

## Testing
Execute `pytest` for the full suite or target a case with `pytest tests/test_trades.py -k exit_priority`. Fast-running synthetic fixtures keep regressions visible; add new cases for every helper or strategy tweak.

## Contributing
Follow the guidelines in `AGENTS.md` for coding style, testing expectations, and pull request checklists before submitting changes.
