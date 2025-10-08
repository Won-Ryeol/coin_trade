from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd

from coin_trade.cli_utils import load_structured, parse_key_value
from coin_trade.io_utils import ensure_output_dir
from coin_trade.signals import DEFAULT_THROTTLE, AutoThrottleConfig, Mode, StrategyParams
from coin_trade.tuning import CandidateConfig, RobustSweepConfig, run_robust_sweep


def _parse_candidates(data: Any | None) -> list[CandidateConfig]:
    if data is None:
        return []
    if isinstance(data, dict) and "candidates" in data:
        data = data["candidates"]
    if not isinstance(data, list):
        raise ValueError("Candidate configuration must be a list of entries")

    candidates: list[CandidateConfig] = []
    for idx, entry in enumerate(data, start=1):
        if not isinstance(entry, dict):
            raise ValueError("Each candidate entry must be a mapping")
        label = str(entry.get("label") or f"candidate_{idx}")
        strategy_overrides = entry.get("strategy")
        throttle_overrides = entry.get("throttle")
        if strategy_overrides is None:
            strategy_overrides = {k: v for k, v in entry.items() if k not in {"label", "throttle"}}
        if not isinstance(strategy_overrides, dict):
            raise ValueError(f"Candidate '{label}' strategy overrides must be a mapping")
        if throttle_overrides is not None and not isinstance(throttle_overrides, dict):
            raise ValueError(f"Candidate '{label}' throttle overrides must be a mapping when provided")
        candidates.append(
            CandidateConfig(
                label=label,
                strategy_overrides=strategy_overrides,
                throttle_overrides=throttle_overrides,
            )
        )
    return candidates


def _apply_overrides(params: StrategyParams, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(params, key):
            raise AttributeError(f"StrategyParams has no attribute '{key}'")
        setattr(params, key, value)


def _apply_throttle_overrides(cfg: AutoThrottleConfig, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"AutoThrottleConfig has no attribute '{key}'")
        setattr(cfg, key, value)


def _render_metric_bar(summary: pd.DataFrame, metric: str, output_path: Path) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if summary.empty or mean_col not in summary.columns:
        return

    labels = summary.index.tolist()
    means = summary[mean_col].to_numpy(dtype=float)
    stds = summary[std_col].fillna(0.0).to_numpy(dtype=float) if std_col in summary.columns else None

    fig, ax = plt.subplots(figsize=(max(6.0, len(labels) * 1.2), 4.0))
    bars = ax.bar(labels, means, color="#4c72b0", yerr=stds, capsize=4 if stds is not None else None)
    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("candidate")
    ax.set_title(f"{metric.replace('_', ' ').title()} (mean +/- std)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    for bar, value in zip(bars, means, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robust hyper-parameter tuning for the coin trade strategy")
    parser.add_argument("--candidates", help="JSON/YAML file describing tuning candidates")
    parser.add_argument("--include-base", action="store_true", help="Include the unmodified baseline in the sweep")
    parser.add_argument("--base-label", default="baseline", help="Label to use for the baseline candidate")
    parser.add_argument(
        "--strategy-set",
        dest="strategy_overrides",
        action="append",
        help="Override base strategy param key=value",
    )
    parser.add_argument(
        "--throttle-set",
        dest="throttle_overrides",
        action="append",
        help="Override base throttle param key=value",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 4242, 2025], help="Random seeds for data sampling")
    parser.add_argument("--sample-k", type=int, default=2, help="Number of CSV files to stitch per sample")
    parser.add_argument("--min-rows", type=int, default=120, help="Minimum rows required per sampled file")
    parser.add_argument("--data-dir", default="data", help="Directory containing price CSV files")
    parser.add_argument("--mode", choices=["production", "research"], default="production")
    parser.add_argument("--fees", type=float, default=0.0005)
    parser.add_argument("--slip", type=float, default=0.0)
    parser.add_argument("--priority", choices=["tp", "sl"], default="tp")
    parser.add_argument("--min-trades", type=int, default=200)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metrics to aggregate (defaults defined in tuning module)",
    )
    parser.add_argument("--output-dir", default="artifacts/tuning")
    parser.add_argument("--artifact-prefix", default="")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = ensure_output_dir(args.output_dir)

    mode_value = cast(Mode, args.mode)
    base_params = StrategyParams()
    base_throttle_template = DEFAULT_THROTTLE[mode_value]
    base_throttle = AutoThrottleConfig(**vars(base_throttle_template))

    cli_strategy_overrides = parse_key_value(args.strategy_overrides)
    if cli_strategy_overrides:
        _apply_overrides(base_params, cli_strategy_overrides)

    cli_throttle_overrides = parse_key_value(args.throttle_overrides)
    if cli_throttle_overrides:
        _apply_throttle_overrides(base_throttle, cli_throttle_overrides)

    candidates = []
    if args.include_base:
        candidates.append(CandidateConfig(label=args.base_label, strategy_overrides={}))

    structured = load_structured(args.candidates)
    candidates.extend(_parse_candidates(structured))

    config = RobustSweepConfig(
        data_dir=args.data_dir,
        seeds=[int(seed) for seed in args.seeds],
        sample_k=args.sample_k,
        min_rows=args.min_rows,
        mode=mode_value,
        min_trades=args.min_trades,
        exit_priority=args.priority,
    )

    records_df, summary_df, samples = run_robust_sweep(
        candidates=candidates,
        base_params=base_params,
        base_throttle=base_throttle,
        config=config,
        fees=args.fees,
        slippage=args.slip,
        metrics=args.metrics,
    )

    prefix = args.artifact_prefix
    records_path = output_dir / f"{prefix}robust_records.csv"
    summary_path = output_dir / f"{prefix}robust_summary.csv"
    samples_path = output_dir / f"{prefix}robust_samples.json"
    chart_path = output_dir / f"{prefix}net_return_bar.png"

    records_df.to_csv(records_path, index=False)
    summary_df.to_csv(summary_path)
    samples_path.write_text(json.dumps(samples, indent=2), encoding="utf-8")

    _render_metric_bar(summary_df, "net_return_pct", chart_path)

    print("\n=== Robust Tuning Summary ===")
    print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nRecords saved to: {records_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Sampled files saved to: {samples_path}")
    if chart_path.exists():
        print(f"Chart saved to: {chart_path}")


if __name__ == "__main__":
    main()
