from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from coin_trade.backtester import BacktestResult, run_backtest
from coin_trade.data import load_random_data
from coin_trade.plotting import render_trades_plot
from coin_trade.signals import AutoThrottleConfig, Mode, StrategyParams, DEFAULT_THROTTLE

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

import warnings
warnings.filterwarnings("ignore")

def parse_key_value(pairs: Optional[Iterable[str]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not pairs:
        return result

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' must be in key=value format")
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() in {"true", "false"}:
            parsed: Any = value.lower() == "true"
        else:
            try:
                if "." in value:
                    parsed = float(value)
                    if isinstance(parsed, float) and parsed.is_integer():
                        parsed = int(parsed)
                else:
                    parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        result[key] = parsed
    return result


def load_structured(path: Optional[str]) -> Optional[Any]:
    if path is None:
        return None
    data_path = Path(path)
    text = data_path.read_text(encoding="utf-8")
    suffix = data_path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML files. Install with 'pip install pyyaml'.")
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported config format for {path}")


def load_price_data(path: str) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(path)

    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path)
    elif data_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(data_path)
    else:
        raise ValueError("Data file must be CSV or Parquet")

    df.columns = [c.lower() for c in df.columns]

    if isinstance(df.index, pd.DatetimeIndex):
        pass
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
        df = df.set_index("timestamp")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=False)
        df = df.set_index("datetime")
    else:
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col], utc=False)
        df = df.set_index(first_col)

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Data file missing required columns: {missing}")

    return df.sort_index()


def apply_strategy_overrides(params: StrategyParams, overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(params, key):
            raise AttributeError(f"StrategyParams has no attribute '{key}'")
        setattr(params, key, value)


def apply_throttle_overrides(cfg: AutoThrottleConfig, overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"AutoThrottleConfig has no attribute '{key}'")
        setattr(cfg, key, value)


def ensure_output_dir(path: str) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def summarize_metrics(result: BacktestResult) -> Dict[str, Any]:
    metrics = result.metrics.metrics
    return {
        "trade_count": metrics.get("trade_count"),
        "trades_per_day": metrics.get("trades_per_day"),
        "win_rate": metrics.get("win_rate"),
        "profit_factor": metrics.get("profit_factor"),
        "net_return_pct": metrics.get("net_return_pct"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "sharpe": metrics.get("sharpe"),
        "exposure": metrics.get("exposure"),
    }


def format_metric(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def save_artifacts(result: BacktestResult, output_dir: Path, *, prefix: str = "") -> None:
    trades_path = output_dir / f"{prefix}trades.csv"
    equity_path = output_dir / f"{prefix}equity.csv"
    metrics_path = output_dir / f"{prefix}metrics.json"

    result.trades.to_csv(trades_path, index=False)
    result.equity.to_csv(equity_path, index=True)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(result.metrics.metrics, fh, indent=2)


def maybe_filter(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]
    return df


def run_single_backtest(
    price_df: pd.DataFrame,
    *,
    params: StrategyParams,
    throttle: AutoThrottleConfig,
    mode: Mode,
    fees: float,
    slip: float,
    priority: str,
    entry_df: Optional[pd.DataFrame],
    min_trades: int,
) -> BacktestResult:
    return run_backtest(
        price_df,
        params=params,
        mode=mode,
        throttle_config=throttle,
        fee_rate=fees,
        slippage_rate=slip,
        exit_priority=priority,  # type: ignore[arg-type]
        entry_df=entry_df,
        min_trades_for_stats=min_trades,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based Bitcoin backtester")
    parser.add_argument("--data-dir", default="data", help="Directory containing 15m CSV files")
    parser.add_argument("--sample-k", type=int, default=1, help="Number of CSV files to sample")
    parser.add_argument("--random-seed", type=int, help="Seed for deterministic sampling")
    parser.add_argument("--entry-data", help="Optional higher-frequency dataset for entry prices")
    parser.add_argument("--mode", choices=["production", "research"], default="production")
    parser.add_argument("--fees", type=float, default=0.0005, help="Per-side fee rate")
    parser.add_argument("--slip", type=float, default=0.0, help="Per-side slippage rate")
    parser.add_argument("--priority", choices=["tp", "sl"], default="tp", help="When both TP/SL hit, exit priority")
    parser.add_argument("--output-dir", default="artifacts", help="Directory to store outputs")
    parser.add_argument("--config", help="JSON or YAML file with strategy overrides")
    parser.add_argument("--sweep", help="JSON/YAML file with a list of override sets for parameter sweep")
    parser.add_argument("--set", dest="strategy_overrides", action="append", help="Override strategy param, e.g. cooldown_bars=3")
    parser.add_argument("--throttle-set", dest="throttle_overrides", action="append", help="Override throttle param, e.g. target_trades_per_day_low=5")
    parser.add_argument("--min-trades", type=int, default=10, help="Minimum trades before reporting win-rate metrics")
    parser.add_argument("--plot-trade-index", type=int, help="Render Plotly for a specific trade index")
    parser.add_argument("--start", help="Filter data from this datetime (inclusive)")
    parser.add_argument("--end", help="Filter data up to this datetime (inclusive)")

    args = parser.parse_args()

    price_df, sampled_paths = load_random_data(
        data_dir=args.data_dir,
        sample_k=args.sample_k,
        random_seed=args.random_seed,
    )
    price_df = maybe_filter(price_df, args.start, args.end)
    if price_df.empty:
        raise ValueError("Filtered price data is empty")

    entry_df = load_price_data(args.entry_data) if args.entry_data else None
    if entry_df is not None:
        entry_df = maybe_filter(entry_df, args.start, args.end)

    sampled_paths_str = [str(p) for p in sampled_paths]

    strategy_params = StrategyParams()
    base_throttle_template = DEFAULT_THROTTLE[args.mode]  # type: ignore[index]
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
            mode=args.mode,  # type: ignore[arg-type]
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
            print(f"(info) trades below threshold {args.min_trades}; win_rate/profit_factor reported as N/A")

        save_artifacts(result, output_dir, prefix=prefix)
        if result.trades.empty:
            print('(info) no trades generated; skipping plot export')
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
            result = run_and_report(params=params_copy, throttle=throttle_copy, label=label, prefix=prefix)
            record = summarize_metrics(result)
            record["label"] = label
            records.append(record)

        if records:
            summary_path = output_dir / "sweep_summary.csv"
            pd.DataFrame(records).to_csv(summary_path, index=False)
    else:
        run_and_report(params=strategy_params, throttle=base_throttle, label="base")


if __name__ == "__main__":
    main()

