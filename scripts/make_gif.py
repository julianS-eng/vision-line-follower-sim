#!/usr/bin/env python3
"""Run one closed-loop simulation and export an annotated GIF of the run.

Usage:
    python scripts/make_gif.py --track curvy_loop --controller stanley

Writes docs/img/run_<track>_<controller>.gif (kept under 5 MB).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vision_line_follower.control.pid import PIDController  # noqa: E402
from vision_line_follower.control.pure_pursuit import PurePursuitController  # noqa: E402
from vision_line_follower.control.stanley import StanleyController  # noqa: E402
from vision_line_follower.sim.camera import CameraConfig, CameraModel  # noqa: E402
from vision_line_follower.sim.robot import RobotConfig  # noqa: E402
from vision_line_follower.sim.world import (  # noqa: E402
    SimulationConfig,
    SpeedSchedule,
    StepTelemetry,
    run_simulation,
)
from vision_line_follower.track.generator import TrackSpec, generate_track  # noqa: E402
from vision_line_follower.track.rendering import TrackRenderConfig, render_track  # noqa: E402
from vision_line_follower.vision.pipeline import VisionPipeline  # noqa: E402
from vision_line_follower.viz.gif_export import GifExportConfig, export_run_gif  # noqa: E402

_CONTROLLERS = {
    "pid": PIDController,
    "pure_pursuit": PurePursuitController,
    "stanley": StanleyController,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track", default="curvy_loop", choices=["oval", "figure_eight", "curvy_loop"]
    )
    parser.add_argument("--controller", default="stanley", choices=list(_CONTROLLERS))
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    track = generate_track(TrackSpec(name=args.track, seed=args.seed, num_crossings=1))
    render_config = TrackRenderConfig(
        lighting_strength=0.3, occlusion_level=0.15, texture_noise_std=7.0, seed=1
    )
    world = render_track(track, render_config)
    camera = CameraModel(CameraConfig(pixel_noise_std=3.0))
    pipeline = VisionPipeline(camera)
    controller = _CONTROLLERS[args.controller]()
    sim_config = SimulationConfig(max_steps=2500, camera_seed=7, robot_seed=7)

    poses = []
    camera_frames = []
    cross_track_errors = []

    record_every = 2

    def on_step(tel: StepTelemetry) -> None:
        if tel.camera_image is not None:
            poses.append(tel.pose)
            camera_frames.append(tel.camera_image)
            cross_track_errors.append(tel.cross_track_error_m)

    result = run_simulation(
        track,
        world,
        camera,
        pipeline,
        controller,
        RobotConfig(),
        SpeedSchedule(),
        sim_config,
        on_step=on_step,
        record_camera_every=record_every,
    )
    print(
        f"track={args.track} controller={args.controller} success={result.success} "
        f"failure_reason={result.failure_reason} laps={result.laps_completed:.2f} "
        f"steps={result.steps_run} mean_err_cm={result.mean_abs_cross_track_error_m * 100:.2f}"
    )

    default_name = f"run_{args.track}_{args.controller}.gif"
    out_path = Path(args.out) if args.out else REPO_ROOT / "docs" / "img" / default_name
    export_run_gif(
        track,
        world,
        poses,
        camera_frames,
        cross_track_errors,
        args.controller,
        out_path,
        GifExportConfig(fps=15, max_frames=220),
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
