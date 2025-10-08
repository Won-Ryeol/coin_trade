from __future__ import annotations

import argparse
import warnings
from collections.abc import Iterable
from typing import Any, cast

from coin_trade.backtester import BacktestResult, run_backtest
from coin_trade.cli_utils import load_structured, parse_key_value
from coin_trade.config import DEFAULT_EXECUTION
from coin_trade.data import load_random_data
from coin_trade.io_utils import ensure_output_dir, load_price_data, maybe_filter, save_artifacts
from coin_trade.plotting import render_trades_plot
from coin_trade.reporting import format_metric, summarize_metrics
from coin_trade.signals import DEFAULT_THROTTLE, AutoThrottleConfig, Mode, StrategyParams

warnings.filterwarnings("ignore")


def apply_strategy_overrides(params: StrategyParams, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(params, key):
            raise AttributeError(f"StrategyParams has no attribute '{key}'")
        setattr(params, key, value)


def apply_throttle_overrides(cfg: AutoThrottleConfig, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"AutoThrottleConfig has no attribute '{key}'")
        setattr(cfg, key, value)


def run_single_backtest(
    price_df,
    *,
    params,
    throttle,
    mode,
    fees,
    slip,
    priority,
    entry_df,
    min_trades,
):
    return run_backtest(
        price_df,
        params=params,
        mode=mode,
        throttle_config=throttle,
        fee_rate=fees,
        slippage_rate=slip,
        exit_priority=priority,
        entry_df=entry_df,
        min_trades_for_stats=min_trades,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rule-based Bitcoin backtester")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing 15m CSV files",
    )
    parser.add_argument(
        "--sample-k",
        type=int,
        default=1,
        help="Number of CSV files to sample",
    )
    parser.add_argument("--random-seed", type=int, help="Seed for deterministic sampling")
    parser.add_argument(
        "--min-rows",
        type=int,
        default=60,
        help="Minimum rows required per sampled file",
    )
    parser.add_argument(
        "--entry-data",
        help="Optional higher-frequency dataset for entry prices",
    )
    parser.add_argument(
        "--mode",
        choices=["production", "research"],
        default="production",
    )
    parser.add_argument(
        "--fees",
        type=float,
        default=DEFAULT_EXECUTION.fee_rate,
        help="Per-side fee rate",
    )
    parser.add_argument(
        "--slip",
        type=float,
        default=DEFAULT_EXECUTION.slippage_rate,
        help="Per-side slippage rate",
    )
    parser.add_argument(
        "--priority",
        choices=["tp", "sl"],
        default="tp",
        help="When both TP/SL hit, exit priority",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory to store outputs",
    )
    parser.add_argument("--config", help="JSON or YAML file with strategy overrides")
    parser.add_argument(
        "--sweep",
        help="JSON/YAML file with a list of override sets for parameter sweep",
    )
    parser.add_argument(
        "--set",
        dest="strategy_overrides",
        action="append",
        help="Override strategy param, for example cooldown_bars=3",
    )
    parser.add_argument(
        "--throttle-set",
        dest="throttle_overrides",
        action="append",
        help="Override throttle param, for example target_trades_per_day_low=5",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=10,
        help="Minimum trades before reporting win-rate metrics",
    )
    parser.add_argument(
        "--plot-trade-index",
        type=int,
        help="Render Plotly for a specific trade index",
    )
    parser.add_argument("--start", help="Filter data from this datetime (inclusive)")
    parser.add_argument("--end", help="Filter data up to this datetime (inclusive)")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    price_df, sampled_paths = load_random_data(
        data_dir=args.data_dir,
        sample_k=args.sample_k,
        random_seed=args.random_seed,
        min_rows=args.min_rows,
    )
    price_df = maybe_filter(price_df, args.start, args.end)
    if price_df.empty:
        raise ValueError("Filtered price data is empty")

    entry_df = load_price_data(args.entry_data) if args.entry_data else None
    if entry_df is not None:
        entry_df = maybe_filter(entry_df, args.start, args.end)

    sampled_paths_str = [str(p) for p in sampled_paths]

    strategy_params = StrategyParams()
    mode_value = cast(Mode, args.mode)
    base_throttle_template = DEFAULT_THROTTLE[mode_value]
    base_throttle = AutoThrottleConfig(**vars(base_throttle_template))

    config_data = load_structured(args.config)
    if isinstance(config_data, dict):
        strategy_overrides = config_data.get("strategy") or {}
        throttle_overrides = config_data.get("throttle") or {}
        if strategy_overrides:
            apply_strategy_overrides(strategy_params, strategy_overrides)
        if throttle_overrides:
            apply_throttle_overrides(base_throttle, throttle_overrides)

    cli_strategy_overrides = parse_key_value(args.strategy_overrides)
    if cli_strategy_overrides:
        apply_strategy_overrides(strategy_params, cli_strategy_overrides)

    cli_throttle_overrides = parse_key_value(args.throttle_overrides)
    if cli_throttle_overrides:
        apply_throttle_overrides(base_throttle, cli_throttle_overrides)

    sweep_data = load_structured(args.sweep)
    output_dir = ensure_output_dir(args.output_dir)

    def run_and_report(
        *,
        params: StrategyParams,
        throttle: AutoThrottleConfig,
        label: str,
        prefix: str = "",
    ) -> BacktestResult:
        params_copy = StrategyParams(**vars(params))
        throttle_copy = AutoThrottleConfig(**vars(throttle))
        result = run_single_backtest(
            price_df,
            params=params_copy,
            throttle=throttle_copy,
            mode=mode_value,
            fees=args.fees,
            slip=args.slip,
            priority=args.priority,
            entry_df=entry_df,
            min_trades=args.min_trades,
        )

        print(f"\n=== {label} ===")
        print("   sampled_files:", ", ".join(sampled_paths_str))
        summary = summarize_metrics(result)
        for key in [
            "trade_count",
            "trades_per_day",
            "win_rate",
            "profit_factor",
            "net_return_pct",
            "max_drawdown_pct",
            "sharpe",
            "exposure",
        ]:
            print(f"{key:>16}: {format_metric(summary[key])}")
        if summary["trade_count"] is not None and summary["trade_count"] < args.min_trades:
            print(f"(info) trades below threshold {args.min_trades}; " "win_rate/profit_factor reported as N/A")

        save_artifacts(result, output_dir, prefix=prefix)
        if result.trades.empty:
            print("(info) no trades generated; skipping plot export")
        else:
            plot_path = output_dir / f"{prefix}backtest.html"
            render_trades_plot(
                result.signals,
                result.trades,
                title=f"Backtest - {label}",
                output_path=str(plot_path),
                trade_index=args.plot_trade_index,
            )
        return result

    if isinstance(sweep_data, list):
        records = []
        for idx, entry in enumerate(sweep_data, start=1):
            label = entry.get("label") or f"config_{idx}"
            overrides = entry.get("strategy") or entry
            throttle_overrides = entry.get("throttle") or {}

            params_copy = StrategyParams(**vars(strategy_params))
            apply_strategy_overrides(params_copy, overrides)

            throttle_copy = AutoThrottleConfig(**vars(base_throttle))
            if throttle_overrides:
                apply_throttle_overrides(throttle_copy, throttle_overrides)

            prefix = f"{label}_".replace(" ", "_")
            result = run_and_report(
                params=params_copy,
                throttle=throttle_copy,
                label=label,
                prefix=prefix,
            )
            record = summarize_metrics(result)
            record["label"] = label
            records.append(record)

        if records:
            import pandas as pd

            summary_path = output_dir / "sweep_summary.csv"
            pd.DataFrame(records).to_csv(summary_path, index=False)
    else:
        run_and_report(params=strategy_params, throttle=base_throttle, label="base")


if __name__ == "__main__":
    main()
