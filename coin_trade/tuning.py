from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .backtester import run_backtest
from .data import load_random_data
from .signals import DEFAULT_THROTTLE, AutoThrottleConfig, Mode, StrategyParams
from .trades import TradeExitPriority


@dataclass(frozen=True)
class CandidateConfig:
    label: str
    strategy_overrides: Mapping[str, object]
    throttle_overrides: Mapping[str, object] | None = None


@dataclass(frozen=True)
class RobustSweepConfig:
    data_dir: str
    seeds: Sequence[int]
    sample_k: int
    min_rows: int
    mode: Mode
    min_trades: int
    exit_priority: TradeExitPriority = "tp"


DEFAULT_METRICS: tuple[str, ...] = (
    "net_return_pct",
    "win_rate",
    "profit_factor",
    "max_drawdown_pct",
    "sharpe",
    "trade_count",
    "trades_per_day",
)


def _apply_strategy_overrides(params: StrategyParams, overrides: Mapping[str, object]) -> None:
    for key, value in overrides.items():
        if not hasattr(params, key):
            raise AttributeError(f"StrategyParams has no attribute '{key}'")
        setattr(params, key, value)


def _apply_throttle_overrides(throttle: AutoThrottleConfig, overrides: Mapping[str, object]) -> None:
    for key, value in overrides.items():
        if not hasattr(throttle, key):
            raise AttributeError(f"AutoThrottleConfig has no attribute '{key}'")
        setattr(throttle, key, value)


def _materialise_params(
    params: StrategyParams,
    throttle: AutoThrottleConfig,
    *,
    strategy_overrides: Mapping[str, object] | None,
    throttle_overrides: Mapping[str, object] | None,
) -> tuple[StrategyParams, AutoThrottleConfig]:
    params_copy = StrategyParams(**vars(params))
    throttle_copy = AutoThrottleConfig(**vars(throttle))
    if strategy_overrides:
        _apply_strategy_overrides(params_copy, dict(strategy_overrides))
    if throttle_overrides:
        _apply_throttle_overrides(throttle_copy, dict(throttle_overrides))
    return params_copy, throttle_copy


def _ensure_candidates(candidates: Iterable[CandidateConfig]) -> list[CandidateConfig]:
    result = list(candidates)
    if not result:
        raise ValueError("At least one candidate configuration is required")
    labels = [candidate.label for candidate in result]
    if len(labels) != len(set(labels)):
        raise ValueError("Candidate labels must be unique")
    return result


def run_robust_sweep(
    *,
    candidates: Iterable[CandidateConfig],
    base_params: StrategyParams | None = None,
    base_throttle: AutoThrottleConfig | None = None,
    config: RobustSweepConfig,
    fees: float,
    slippage: float,
    metrics: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, list[str]]]:
    """Run each candidate across a shared set of data samples and aggregate metrics."""
    if not config.seeds:
        raise ValueError("At least one seed is required for robust sweeps")

    materialised_candidates = _ensure_candidates(candidates)
    params_template = base_params or StrategyParams()
    throttle_template = base_throttle or DEFAULT_THROTTLE[config.mode]

    sampled_data: dict[int, tuple[pd.DataFrame, list[Path]]] = {}
    for seed in config.seeds:
        price_df, paths = load_random_data(
            config.data_dir,
            sample_k=config.sample_k,
            random_seed=int(seed),
            min_rows=config.min_rows,
        )
        sampled_data[int(seed)] = (price_df, paths)

    records: list[dict[str, object]] = []

    for candidate in materialised_candidates:
        for seed, (price_df, paths) in sampled_data.items():
            params_copy, throttle_copy = _materialise_params(
                params_template,
                throttle_template,
                strategy_overrides=candidate.strategy_overrides,
                throttle_overrides=candidate.throttle_overrides,
            )

            result = run_backtest(
                price_df,
                params=params_copy,
                mode=config.mode,
                throttle_config=throttle_copy,
                fee_rate=fees,
                slippage_rate=slippage,
                exit_priority=config.exit_priority,
                min_trades_for_stats=config.min_trades,
            )

            record = {
                **{key: float(value) for key, value in result.metrics.metrics.items()},
                "label": candidate.label,
                "seed": seed,
                "sampled_files": "|".join(str(p) for p in paths),
                "trade_count_actual": float(len(result.trades)),
            }
            records.append(record)

    records_df = pd.DataFrame.from_records(records)
    if records_df.empty:
        raise RuntimeError("Robust sweep produced no records; check candidate definitions")

    metric_list = tuple(metrics) if metrics else DEFAULT_METRICS
    summary_df = summarise_records(records_df, metric_list)
    samples = {seed: [str(p) for p in paths] for seed, (_, paths) in sampled_data.items()}

    return records_df, summary_df, samples


def summarise_records(records: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Aggregate candidate performance with stability diagnostics."""
    if records.empty:
        return pd.DataFrame()

    grouped = records.groupby("label")
    available_metrics = [metric for metric in metrics if metric in records.columns]

    if not available_metrics:
        raise ValueError("None of the requested metrics are present in the records")

    agg_map: dict[str, list[str]] = {metric: ["mean", "std", "min", "max"] for metric in available_metrics}
    aggregated = grouped.agg(agg_map)

    flattened_columns = [f"{metric}_{stat}" for metric, stat in aggregated.columns]
    aggregated.columns = flattened_columns
    aggregated["runs"] = grouped.size()

    if "net_return_pct_mean" in aggregated.columns and "net_return_pct_std" in aggregated.columns:
        aggregated["stability_score"] = aggregated["net_return_pct_mean"] / aggregated[
            "net_return_pct_std"
        ].replace(0.0, np.nan)

    if "max_drawdown_pct_mean" in aggregated.columns:
        aggregated["average_drawdown"] = aggregated["max_drawdown_pct_mean"].abs()

    if "win_rate_mean" in aggregated.columns:
        aggregated["win_rate_cv"] = aggregated["win_rate_std"] / aggregated["win_rate_mean"].replace(0.0, np.nan)

    aggregated = aggregated.sort_values(by="net_return_pct_mean", ascending=False, na_position="last")
    return aggregated
