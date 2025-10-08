# Coin Trade Backtester

## Overview
Coin Trade is a Bitcoin-focused backtesting harness that samples 15?min OHLCV data, applies regime-aware squeeze/trend entry logic, and exports trades, equity curves, and plots for strategy review.

## Architecture
- coin_trade/config.py centralises execution costs, ATR/ADX windows, and default buffers.
- coin_trade/utils/ provides reusable dataframe hygiene helpers (dataframe.py) and R/return math (execution.py).
- Trading flow lives in signals.py (signal generation), 	rades.py (order lifecycle), and acktester.py (orchestration).
- Reporting and IO helpers live in eporting.py, io_utils.py, and plotting utilities in plotting.py.
- Tests cover loaders, regime detection, execution maths, and trade edges under 	ests/.

## Tooling
Use the project interpreter C:\Users\wrk92\anaconda3\envs\coin_dest\python.exe for every command.

`ash
# Lint
C:\Users\wrk92\anaconda3\envs\coin_dest\python.exe -m ruff check

# Type check
C:\Users\wrk92\anaconda3\envs\coin_dest\python.exe -m mypy .

# Test with coverage
C:\Users\wrk92\anaconda3\envs\coin_dest\python.exe -m pytest
`

## Running Backtests
`ash
C:\Users\wrk92\anaconda3\envs\coin_dest\python.exe run_backtest.py --mode research --sample-k 3 --min-rows 60
C:\Users\wrk92\anaconda3\envs\coin_dest\python.exe run_backtest.py --sweep sweep.yaml --sample-k 2
`
Both commands emit trades, equity, and metrics into rtifacts/. Override parameters with repeated --set key=value or throttle knobs via --throttle-set key=value.

## Data Expectations
- Place source CSVs beneath data/; frequencies should be specified with the min suffix (e.g., 15min).
- Generated artifacts accumulate under rtifacts/ for manual inspection.

## Testing
Run the full suite with the pytest command above. Synthetic fixtures exercise regime detection, trailing stops, partial profit logic, and dataframe hygiene. Coverage is collected automatically; add targeted tests whenever you modify core modules or utilities.

## Change Log & Guides
- See CHANGELOG.md for a summary of refactors, removals, and module moves.
- AGENTS.md documents contributor workflows, including interpreter usage, lint/type/test commands, and module layout.
## Robust Tuning
```bash
C:\Users\wrk92\anaconda3\envs\coin_dest\python.exe run_tuning.py --candidates sweep.yaml --include-base --seeds 42 4242 2025 --sample-k 2 --output-dir artifacts/tuning
```
run_tuning.py reuses the same sampled OHLCV blocks for every candidate, aggregates win rate, profit factor, sharpe, drawdowns, and exports CSV/PNG summaries in artifacts/tuning/ for easy comparison.

## Risk-Reward Configuration
- StrategyParams exposes rr_* and sl_* schedule knobs to shape breakout/reversion targets without rewriting signal logic.
- coin_trade/risk_reward.py handles interpolation, scaling, and buffers so tweaks remain declarative and testable.
- rr_breakout/rr_reversion columns are now stamped on the signal frame to aid diagnostics and charting during sweeps.
