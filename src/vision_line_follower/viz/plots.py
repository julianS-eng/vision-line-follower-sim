"""Matplotlib figures summarizing benchmark results, saved to ``docs/img/``."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vision_line_follower.benchmark.run_benchmark import AggregateRow, BenchmarkRecord
from vision_line_follower.track.generator import Track

_COLORS = {"pid": "#1f77b4", "pure_pursuit": "#ff7f0e", "stanley": "#2ca02c"}
_LABELS = {"pid": "PID", "pure_pursuit": "Pure Pursuit", "stanley": "Stanley"}


def plot_mean_error_bars(rows: list[AggregateRow], path: Path) -> None:
    """Grouped bar chart: mean |cross-track error| per controller x noise level."""
    controllers = sorted({r.controller_name for r in rows})
    noise_levels = sorted(
        {r.noise_level for r in rows}, key=lambda n: ["clean", "moderate", "harsh"].index(n)
    )

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    n_groups = len(noise_levels)
    n_bars = len(controllers)
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    by_key = {(r.controller_name, r.noise_level): r for r in rows}
    for i, controller in enumerate(controllers):
        values = [
            by_key[(controller, nl)].mean_abs_error_m * 100 if (controller, nl) in by_key else 0.0
            for nl in noise_levels
        ]
        ax.bar(
            x + i * width - 0.4 + width / 2,
            values,
            width=width,
            label=_LABELS.get(controller, controller),
            color=_COLORS.get(controller),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in noise_levels])
    ax.set_ylabel("Mean |cross-track error| (cm)")
    ax.set_xlabel("Noise level")
    ax.set_title("Controller accuracy vs. noise level")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_success_rate_bars(rows: list[AggregateRow], path: Path) -> None:
    """Grouped bar chart: success rate per controller x noise level."""
    controllers = sorted({r.controller_name for r in rows})
    noise_levels = sorted(
        {r.noise_level for r in rows}, key=lambda n: ["clean", "moderate", "harsh"].index(n)
    )

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    n_groups = len(noise_levels)
    n_bars = len(controllers)
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    by_key = {(r.controller_name, r.noise_level): r for r in rows}
    for i, controller in enumerate(controllers):
        values = [
            by_key[(controller, nl)].success_rate * 100 if (controller, nl) in by_key else 0.0
            for nl in noise_levels
        ]
        ax.bar(
            x + i * width - 0.4 + width / 2,
            values,
            width=width,
            label=_LABELS.get(controller, controller),
            color=_COLORS.get(controller),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in noise_levels])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Noise level")
    ax.set_title("Lap-completion success rate vs. noise level")
    ax.legend(frameon=False, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_error_boxplot(records: list[BenchmarkRecord], path: Path) -> None:
    """Boxplot of per-run mean |cross-track error|, grouped by controller."""
    by_controller: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_controller[r.controller_name].append(r.mean_abs_cross_track_error_m * 100)

    controllers = sorted(by_controller)
    data = [by_controller[c] for c in controllers]

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
    bplot = ax.boxplot(
        data, tick_labels=[_LABELS.get(c, c) for c in controllers], patch_artist=True
    )
    for patch, controller in zip(bplot["boxes"], controllers, strict=True):
        patch.set_facecolor(_COLORS.get(controller, "#888888"))
        patch.set_alpha(0.6)
    ax.set_ylabel("Mean |cross-track error| per run (cm)")
    ax.set_title("Error distribution across all benchmark runs")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_trajectory(track: Track, trajectories: dict[str, np.ndarray], path: Path) -> None:
    """Overlay controller trajectories (each ``(N, 3)`` pose array) on the track."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    closed = np.vstack([track.centerline, track.centerline[0]])
    ax.plot(
        closed[:, 0], closed[:, 1], color="#333333", linewidth=3, label="Track centerline", zorder=1
    )

    for name, poses in trajectories.items():
        ax.plot(
            poses[:, 0],
            poses[:, 1],
            color=_COLORS.get(name),
            linewidth=1.6,
            label=_LABELS.get(name, name),
            zorder=2,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Closed-loop trajectories -- {track.name.replace('_', ' ')}")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
