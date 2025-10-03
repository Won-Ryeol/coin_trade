# Coin Trade Backtester

Codex-managed Bitcoin backtester focused on realistic order handling, configurable risk controls, and Plotly visualization.

## Highlights
- Bollinger-Keltner squeeze breakout with optional RSI mean-reversion and low-risk helper.
- Dynamic ATR-based TP/SL sizing with ADX-conditioned risk/reward and fee/slippage buffers.
- Higher-timeframe trend and daily-return filters plus intraday loss caps to avoid weak breakout regimes.
- Next-bar open entries, intrabar TP/SL detection, no look-ahead bias, and single-position enforcement.
- CLI auto-discovers 15m CSVs under `data/`, samples inputs per run, prints key metrics, saves `trades.csv`, `equity.csv`, and Plotly HTML.
- Parameter sweep utility (JSON/YAML) and minimum-trade guard before reporting win-rate stats.
- Pytest suite using synthetic OHLCV to cover trade construction edge cases and signal gating.

## Setup
Requires Python 3.8+. Optional: install PyYAML for YAML configs.

```bash
pip install pandas numpy plotly pytest
pip install pyyaml  # optional, for YAML configs
```

## Basic Usage
Place one or more 15-minute OHLCV CSVs under `data/`, then run:

```bash
python run_backtest.py --mode research --sample-k 3 --random-seed 42 --min-rows 60 --fees 0.0005 --slip 0.0002
```

Outputs (default `artifacts/`): `trades.csv`, `equity.csv`, `metrics.json`, and `backtest.html` (skipped if no trades).

### Common Flags
- `--sample-k 5` ? sample this many CSVs from the `data/` tree (default 1).
- `--min-rows 80` ? require at least this many rows in each sampled CSV before it is used.
- `--random-seed 42` ? keep file sampling deterministic run-to-run.
- `--data-dir alt_data` ? point to another folder of 15m CSVs.
- `--entry-data btcusdt_5m.csv` ? optional higher-frequency dataset for next-bar entries.
- `--set key=value` ? override strategy param (repeatable), e.g. `--set cooldown_bars=2 --set max_stopouts_per_day=2`.
- `--throttle-set key=value` ? override auto-throttle settings, e.g. `--throttle-set target_trades_per_day_low=4`.
- `--config config.yaml` ? structured overrides (`strategy` and/or `throttle` sections; JSON also supported).
- `--sweep sweep.yaml` ? list of override sets; saves per-run artifacts plus `sweep_summary.csv`.
- `--plot-trade-index 3` ? render Plotly view centered on trade #3.
- `--min-trades 200` ? threshold before reporting win rate / profit factor (metrics show `N/A` below this count).
- `--start 2024-01-01 --end 2024-06-01` ? optional slicing window.

## Parameter Sweep Example (`sweep.yaml`)
```yaml
- label: relaxed_cooldown
  strategy:
    cooldown_bars: 1
    squeeze_pct: 0.55
  throttle:
    target_trades_per_day_low: 8
    target_trades_per_day_high: 14
- label: strict_adx
  strategy:
    adx_threshold: 28
    volume_multiplier: 1.35
```
Run:
```bash
python run_backtest.py --sweep sweep.yaml --sample-k 2 --random-seed 7
```

## Testing
```bash
pytest
```
Covers next-open entries, TP/SL tie-breaking, signal gating (cooldown/per-day), fee/slippage math, and warm-up behaviour.

## Notes
- Data loader randomly samples CSVs under the configured directory each run, skipping empty or short files automatically, and prints the selected file list.
- Higher-timeframe trend context plus daily-return and intraday stopout/loss caps are configurable via `--set ema_long_len=128 --set max_stopouts_per_day=2`.
- Auto-throttle keeps trade frequency within mode-specific targets by nudging cooldown/per-day caps and sensitivity thresholds.
- `coin_trade/signals.py` returns legacy `buy_signal`, `tp_pct`, `sl_pct` columns for easy integration.
- Plotly HTML uses CDN script; view locally via browser.
