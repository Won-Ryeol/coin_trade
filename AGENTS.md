# Repository Guidelines

## Project Structure & Module Organization
The `coin_trade/` package contains the backtester pipeline: `signals.py` orchestrates squeeze logic, `trades.py` builds fills, and `metrics.py` plus `plotting.py` format outputs. CLI helpers live in `run_backtest.py` and `cli_utils.py`. Tests under `tests/` mirror modules (`test_trades.py`, `test_signals.py`). Place raw 15m CSV data beneath `data/`; generated artifacts land in `artifacts/`.

## Build, Test, and Development Commands
- `python -m venv .venv && .venv\Scripts\activate` prepares a local environment.
- `pip install pandas numpy plotly pytest pyyaml` installs runtime and test deps.
- `python run_backtest.py --mode research --sample-k 1` executes a single-run backtest and writes `artifacts/`.
- `python run_backtest.py --sweep sweep.yaml --sample-k 2` sweeps override sets defined in `sweep.yaml`.
- `pytest` runs all unit tests; use `pytest tests/test_trades.py -k exit_priority` when iterating on a scenario.

## Coding Style & Naming Conventions
Use Python 3.8+ with 4-space indentation and type hints (`BacktestResult`, `StrategyParams`). Keep public functions documented with one-line docstrings and prefer explicit keyword-only arguments for clarity. Follow the existing snake_case for functions, snake_case CSV file names, and uppercase constants. Favor small, composable helpers rather than expanding monoliths like `run_backtest`.

## Testing Guidelines
Pytest drives coverage; every new helper should have a deterministic fixture in `tests/conftest.py`. Name test files `test_<module>.py` and keep focused parametrized cases to surface regressions (ADX thresholds, TP/SL tie-breaking, cooldown caps). Ensure new strategies hit the minimum trades guard; capture regression datasets in `data/samples/` when needed. Backtests run quickly, so add at least one assertion on metrics for each new config.

## Commit & Pull Request Guidelines
Commit history favors short, imperative subjects (`refactor`, `init`). Continue that style and optionally append a scope (`signals`) when touching a single module. For pull requests, describe the strategy change, list commands run (`pytest`, sample backtest), and attach the resulting `artifacts/metrics.json` diff or Plotly screenshot. Link open issues or TODOs and call out any configuration flags that now default differently.
