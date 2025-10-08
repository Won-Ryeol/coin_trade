# Repository Guidelines

## Project Structure & Module Organization
- coin_trade/config.py defines shared execution and window defaults consumed across signals and trades.
- coin_trade/utils/ hosts reusable dataframe hygiene (dataframe.py) and execution math (execution.py).
- Core engines live in coin_trade/signals.py, coin_trade/trades.py, coin_trade/backtester.py, with IO helpers in io_utils.py and reporting in eporting.py.
- Tests mirror the new utilities and behaviours under 	ests/ (e.g., 	est_utils_execution.py, 	est_regime.py).

## Build, Test, and Development Commands
- Always use C:\Users\wrk92\anaconda3\envs\coin_dest\python.exe for all Python and tooling commands.
- C:\Users\wrk92\anaconda3\envs\coin_dest\python.exe -m ruff check enforces lint rules.
- C:\Users\wrk92\anaconda3\envs\coin_dest\python.exe -m mypy . runs static type checks.
- C:\Users\wrk92\anaconda3\envs\coin_dest\python.exe -m pytest executes the suite with coverage.

## Coding Style & Naming Conventions
- Prefer typed function signatures and reusable helpers in coin_trade/utils over inline duplication.
- Keep frequency literals in ISO form (15min) to avoid deprecation warnings.
- Strategy-level constants derive from config.DEFAULT_EXECUTION and DEFAULT_SIGNAL_WINDOWS; avoid hard-coding fees or ATR windows.

## Testing Guidelines
- Extend synthetic fixtures in 	ests/ when introducing new risk logic or utilities.
- Tests treat warnings as errors; update fixtures or code to avoid deprecated pandas idioms.
- Coverage runs with pytest --cov=coin_trade; add focused unit tests when refactoring core modules.

## Commit & Pull Request Guidelines
- Reference lint/type/test commands above in PR descriptions.
- Note structural moves (e.g., new utility modules) and document any migrations in CHANGELOG.md.
- Keep behavioural changes backward compatible; document parameter or output adjustments explicitly.
