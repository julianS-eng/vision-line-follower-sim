"""Plotting and GIF export helpers for benchmark results and demo runs."""

from vision_line_follower.viz.gif_export import GifExportConfig, export_run_gif
from vision_line_follower.viz.plots import (
    plot_error_boxplot,
    plot_mean_error_bars,
    plot_success_rate_bars,
    plot_trajectory,
)

__all__ = [
    "GifExportConfig",
    "export_run_gif",
    "plot_error_boxplot",
    "plot_mean_error_bars",
    "plot_success_rate_bars",
    "plot_trajectory",
]
