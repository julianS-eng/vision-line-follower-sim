#!/usr/bin/env python3
"""Run a single closed-loop simulation and print a summary.

Usage:
    python scripts/run_demo.py --track oval --controller pid
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
    run_simulation,
)
from vision_line_follower.track.generator import TrackSpec, generate_track  # noqa: E402
from vision_line_follower.track.rendering import TrackRenderConfig, render_track  # noqa: E402
from vision_line_follower.vision.pipeline import VisionPipeline  # noqa: E402

_CONTROLLERS = {
    "pid": PIDController,
    "pure_pursuit": PurePursuitController,
    "stanley": StanleyController,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="oval", choices=["oval", "figure_eight", "curvy_loop"])
    parser.add_argument("--controller", default="pid", choices=list(_CONTROLLERS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    track = generate_track(TrackSpec(name=args.track, seed=args.seed, num_crossings=1))
    render_config = TrackRenderConfig(lighting_strength=0.3, occlusion_level=0.1, seed=1)
    world = render_track(track, render_config)
    camera = CameraModel(CameraConfig())
    pipeline = VisionPipeline(camera)
    controller = _CONTROLLERS[args.controller]()

    result = run_simulation(
        track,
        world,
        camera,
        pipeline,
        controller,
        RobotConfig(),
        SpeedSchedule(),
        SimulationConfig(max_steps=3000),
    )

    print(f"Track:            {result.track_name}")
    print(f"Controller:       {result.controller_name}")
    print(f"Success:          {result.success} (reason: {result.failure_reason})")
    print(f"Laps completed:   {result.laps_completed:.2f}")
    print(f"Sim time:         {result.sim_time_s:.1f} s ({result.steps_run} steps)")
    print(f"Mean |error|:     {result.mean_abs_cross_track_error_m * 100:.2f} cm")
    print(f"Max |error|:      {result.max_abs_cross_track_error_m * 100:.2f} cm")
    print(f"RMS error:        {result.rms_cross_track_error_m * 100:.2f} cm")
    print(f"Mean speed:       {result.mean_speed_mps:.3f} m/s")
    print(f"Vision found rate:{result.vision_found_rate * 100:.1f}%")


if __name__ == "__main__":
    main()
