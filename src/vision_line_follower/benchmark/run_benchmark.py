"""Multi-track, multi-controller, multi-noise-level benchmark.

Every combination of track shape, difficulty ("noise") level and controller
is run for several random seeds so that reported metrics are averages over
independent trials, not single lucky/unlucky runs. All metrics reported in
the README are produced by actually executing this benchmark -- see
``scripts/run_benchmark.py``.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

from vision_line_follower.control.base import Controller
from vision_line_follower.control.pid import PIDConfig, PIDController
from vision_line_follower.control.pure_pursuit import PurePursuitConfig, PurePursuitController
from vision_line_follower.control.stanley import StanleyConfig, StanleyController
from vision_line_follower.sim.camera import CameraConfig, CameraModel
from vision_line_follower.sim.robot import RobotConfig
from vision_line_follower.sim.world import (
    SimulationConfig,
    SimulationResult,
    SpeedSchedule,
    run_simulation,
)
from vision_line_follower.track.generator import TrackName, TrackSpec, generate_track
from vision_line_follower.track.rendering import TrackRenderConfig, render_track
from vision_line_follower.vision.pipeline import VisionPipeline


@dataclass(frozen=True, slots=True)
class NoiseLevel:
    """A named difficulty preset bundling render/camera/robot noise knobs."""

    name: str
    lighting_strength: float
    occlusion_level: float
    texture_noise_std: float
    pixel_noise_std: float
    wheel_speed_noise_std: float


DEFAULT_NOISE_LEVELS: tuple[NoiseLevel, ...] = (
    NoiseLevel(
        "clean",
        lighting_strength=0.10,
        occlusion_level=0.0,
        texture_noise_std=4.0,
        pixel_noise_std=2.0,
        wheel_speed_noise_std=0.0,
    ),
    NoiseLevel(
        "moderate",
        lighting_strength=0.35,
        occlusion_level=0.15,
        texture_noise_std=8.0,
        pixel_noise_std=5.0,
        wheel_speed_noise_std=0.008,
    ),
    NoiseLevel(
        "harsh",
        lighting_strength=0.55,
        occlusion_level=0.35,
        texture_noise_std=14.0,
        pixel_noise_std=9.0,
        wheel_speed_noise_std=0.02,
    ),
)

DEFAULT_TRACK_NAMES: tuple[TrackName, ...] = ("oval", "figure_eight", "curvy_loop")


def _build_controller(name: str) -> Controller:
    if name == "pid":
        return PIDController(PIDConfig())
    if name == "pure_pursuit":
        return PurePursuitController(PurePursuitConfig())
    if name == "stanley":
        return StanleyController(StanleyConfig())
    raise ValueError(f"Unknown controller: {name}")


DEFAULT_CONTROLLER_NAMES: tuple[str, ...] = ("pid", "pure_pursuit", "stanley")


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Top-level knobs for the benchmark sweep.

    Attributes:
        track_names: Track shapes to evaluate.
        controller_names: Controllers to evaluate.
        noise_levels: Difficulty presets to evaluate.
        seeds_per_case: Number of independent random seeds per
            (track, noise level, controller) combination.
        track_seed_base: Base seed for track geometry generation (one track
            instance is generated per track name and reused across
            controllers/noise levels/seeds, so all controllers are compared
            on literally the same track).
        max_steps: Safety cap on simulation steps per run.
    """

    track_names: tuple[TrackName, ...] = DEFAULT_TRACK_NAMES
    controller_names: tuple[str, ...] = DEFAULT_CONTROLLER_NAMES
    noise_levels: tuple[NoiseLevel, ...] = DEFAULT_NOISE_LEVELS
    seeds_per_case: int = 3
    track_seed_base: int = 100
    max_steps: int = 3000


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    track_name: str
    noise_level: str
    controller_name: str
    run_seed: int


@dataclass(slots=True)
class BenchmarkRecord:
    """Flattened, CSV-friendly summary of one benchmark run."""

    track_name: str
    noise_level: str
    controller_name: str
    run_seed: int
    success: bool
    failure_reason: str | None
    laps_completed: float
    sim_time_s: float
    mean_abs_cross_track_error_m: float
    max_abs_cross_track_error_m: float
    rms_cross_track_error_m: float
    mean_speed_mps: float
    vision_found_rate: float

    @staticmethod
    def from_result(case: BenchmarkCase, result: SimulationResult) -> BenchmarkRecord:
        return BenchmarkRecord(
            track_name=case.track_name,
            noise_level=case.noise_level,
            controller_name=case.controller_name,
            run_seed=case.run_seed,
            success=result.success,
            failure_reason=result.failure_reason,
            laps_completed=result.laps_completed,
            sim_time_s=result.sim_time_s,
            mean_abs_cross_track_error_m=result.mean_abs_cross_track_error_m,
            max_abs_cross_track_error_m=result.max_abs_cross_track_error_m,
            rms_cross_track_error_m=result.rms_cross_track_error_m,
            mean_speed_mps=result.mean_speed_mps,
            vision_found_rate=result.vision_found_rate,
        )


def build_default_cases(config: BenchmarkConfig) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for track_name in config.track_names:
        for noise in config.noise_levels:
            for controller_name in config.controller_names:
                for seed_idx in range(config.seeds_per_case):
                    cases.append(
                        BenchmarkCase(
                            track_name=track_name,
                            noise_level=noise.name,
                            controller_name=controller_name,
                            run_seed=seed_idx,
                        )
                    )
    return cases


def run_benchmark(config: BenchmarkConfig | None = None) -> list[BenchmarkRecord]:
    """Run every (track, noise level, controller, seed) combination."""
    cfg = config or BenchmarkConfig()
    records: list[BenchmarkRecord] = []

    for t_idx, track_name in enumerate(cfg.track_names):
        track_spec = TrackSpec(name=track_name, seed=cfg.track_seed_base + t_idx, num_crossings=1)
        track = generate_track(track_spec)

        for noise in cfg.noise_levels:
            render_config = TrackRenderConfig(
                lighting_strength=noise.lighting_strength,
                occlusion_level=noise.occlusion_level,
                texture_noise_std=noise.texture_noise_std,
                seed=cfg.track_seed_base + t_idx,
            )
            world = render_track(track, render_config)
            camera = CameraModel(CameraConfig(pixel_noise_std=noise.pixel_noise_std))
            robot_config = RobotConfig(wheel_speed_noise_std=noise.wheel_speed_noise_std)

            for controller_name in cfg.controller_names:
                for seed_idx in range(cfg.seeds_per_case):
                    case = BenchmarkCase(
                        track_name=track_name,
                        noise_level=noise.name,
                        controller_name=controller_name,
                        run_seed=seed_idx,
                    )
                    pipeline = VisionPipeline(camera)
                    controller = _build_controller(controller_name)
                    sim_config = SimulationConfig(
                        max_steps=cfg.max_steps,
                        camera_seed=1000 + seed_idx,
                        robot_seed=2000 + seed_idx,
                    )
                    result = run_simulation(
                        track,
                        world,
                        camera,
                        pipeline,
                        controller,
                        robot_config,
                        SpeedSchedule(),
                        sim_config,
                    )
                    records.append(BenchmarkRecord.from_result(case, result))
    return records


@dataclass(slots=True)
class AggregateRow:
    controller_name: str
    noise_level: str
    track_name: str
    n_runs: int
    success_rate: float
    mean_abs_error_m: float
    max_abs_error_m: float
    mean_speed_mps: float


def aggregate_results(
    records: list[BenchmarkRecord], group_by: tuple[str, ...] = ("controller_name", "noise_level")
) -> list[AggregateRow]:
    """Aggregate per-run records into per-group summary rows."""
    groups: dict[tuple[str, ...], list[BenchmarkRecord]] = {}
    for r in records:
        key = tuple(getattr(r, field) for field in group_by)
        groups.setdefault(key, []).append(r)

    rows: list[AggregateRow] = []
    for key, group_records in sorted(groups.items()):
        values = dict(zip(group_by, key, strict=True))
        rows.append(
            AggregateRow(
                controller_name=values.get("controller_name", "*"),
                noise_level=values.get("noise_level", "*"),
                track_name=values.get("track_name", "*"),
                n_runs=len(group_records),
                success_rate=statistics.mean(1.0 if r.success else 0.0 for r in group_records),
                mean_abs_error_m=statistics.mean(
                    r.mean_abs_cross_track_error_m for r in group_records
                ),
                max_abs_error_m=max(r.max_abs_cross_track_error_m for r in group_records),
                mean_speed_mps=statistics.mean(r.mean_speed_mps for r in group_records),
            )
        )
    return rows


def write_records_csv(records: list[BenchmarkRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BenchmarkRecord.__dataclass_fields__)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: getattr(r, k) for k in fieldnames})


def write_aggregate_csv(rows: list[AggregateRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(AggregateRow.__dataclass_fields__)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: getattr(r, k) for k in fieldnames})


__all__ = [
    "DEFAULT_CONTROLLER_NAMES",
    "DEFAULT_NOISE_LEVELS",
    "DEFAULT_TRACK_NAMES",
    "AggregateRow",
    "BenchmarkCase",
    "BenchmarkConfig",
    "BenchmarkRecord",
    "NoiseLevel",
    "aggregate_results",
    "build_default_cases",
    "run_benchmark",
    "write_aggregate_csv",
    "write_records_csv",
]
