#!/usr/bin/env python3
"""Run the full benchmark sweep and write CSV results + comparison figures.

Usage:
    python scripts/run_benchmark.py

Writes:
    docs/benchmark_results.csv           -- one row per individual run
    docs/benchmark_summary.csv           -- aggregated by controller x noise level
    docs/img/benchmark_mean_error.png
    docs/img/benchmark_success_rate.png
    docs/img/benchmark_error_boxplot.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vision_line_follower.benchmark.run_benchmark import (  # noqa: E402
    BenchmarkConfig,
    aggregate_results,
    run_benchmark,
    write_aggregate_csv,
    write_records_csv,
)
from vision_line_follower.viz.plots import (  # noqa: E402
    plot_error_boxplot,
    plot_mean_error_bars,
    plot_success_rate_bars,
)


def main() -> None:
    docs_dir = REPO_ROOT / "docs"
    img_dir = docs_dir / "img"

    config = BenchmarkConfig(seeds_per_case=4, max_steps=2500)
    n_cases = (
        len(config.track_names)
        * len(config.controller_names)
        * len(config.noise_levels)
        * config.seeds_per_case
    )
    print(
        f"Running {n_cases} simulations "
        f"({len(config.track_names)} tracks x {len(config.controller_names)} controllers x "
        f"{len(config.noise_levels)} noise levels x {config.seeds_per_case} seeds)..."
    )

    t0 = time.time()
    records = run_benchmark(config)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed / len(records):.2f}s/run average).")

    write_records_csv(records, docs_dir / "benchmark_results.csv")

    summary_rows = aggregate_results(records, group_by=("controller_name", "noise_level"))
    write_aggregate_csv(summary_rows, docs_dir / "benchmark_summary.csv")

    plot_mean_error_bars(summary_rows, img_dir / "benchmark_mean_error.png")
    plot_success_rate_bars(summary_rows, img_dir / "benchmark_success_rate.png")
    plot_error_boxplot(records, img_dir / "benchmark_error_boxplot.png")

    print("\n=== Summary (controller x noise level) ===")
    header = (
        f"{'controller':<14}{'noise':<10}{'n':>4}{'success%':>10}"
        f"{'mean_err_cm':>13}{'max_err_cm':>12}{'mean_v_mps':>12}"
    )
    print(header)
    for row in summary_rows:
        print(
            f"{row.controller_name:<14}{row.noise_level:<10}{row.n_runs:>4}"
            f"{row.success_rate * 100:>9.1f}%{row.mean_abs_error_m * 100:>12.2f} "
            f"{row.max_abs_error_m * 100:>11.2f} {row.mean_speed_mps:>11.3f}"
        )

    print(f"\nWrote {docs_dir / 'benchmark_results.csv'}")
    print(f"Wrote {docs_dir / 'benchmark_summary.csv'}")
    print(f"Wrote figures to {img_dir}")


if __name__ == "__main__":
    main()
