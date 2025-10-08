# Changelog

## Unreleased
- Introduced shared configuration and utility modules (coin_trade.config, coin_trade.utils) to centralise execution costs, price-frame hygiene, and R calculations.
- Refactored trade builder to support partial profit logic via shared helpers while removing the unused TradeBuilderConfig type.
- Refined risk-reward scheduling via coin_trade/risk_reward.py with smooth interpolation and diagnostic columns.
- Added run_tuning.py and tuning utilities for multi-sample parameter sweeps with aggregated stability metrics.
- Documented the robust tuning workflow and exposed new StrategyParams rr/sl schedule knobs.
- Normalised data-loading and IO flows onto the new dataframe utilities for consistent column/index handling.
- Added Ruff, mypy, and pytest coverage configuration via pyproject.toml.
- Added execution utility tests and enriched existing trade behaviour tests to cover partial TP and trailing stops.
