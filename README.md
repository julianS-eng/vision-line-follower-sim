# Vision Line Follower

[![CI](https://github.com/julianS-eng/vision-line-follower-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/julianS-eng/vision-line-follower-sim/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

A simulated differential-drive robot that follows a painted line using a
**classical computer-vision pipeline** (OpenCV) -- no hardware, no deep
learning. Procedurally generated tracks, a physically-motivated pinhole
camera model, a bird's-eye/adaptive-threshold/sliding-window/polynomial-fit
vision pipeline, and three steering controllers (PID, Pure Pursuit, Stanley)
compared head-to-head on a reproducible benchmark.

Lee esto en español: [README.es.md](README.es.md) · Theory, design
rationale and interview Q&A (in Spanish): [docs/LEARNING.md](docs/LEARNING.md)

<p align="center">
  <img src="docs/img/run_curvy_loop_stanley.gif" width="520" alt="Closed-loop run: Stanley controller on the curvy_loop track, camera view inset top-right" />
</p>

## Table of contents

- [Why this project](#why-this-project)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [The vision pipeline](#the-vision-pipeline)
- [The three controllers](#the-three-controllers)
- [Benchmark results](#benchmark-results)
- [Project structure](#project-structure)
- [Testing & CI](#testing--ci)
- [Known limitations & future work](#known-limitations--future-work)
- [License](#license)

## Why this project

Real line-follower robots are cheap to build but slow and fiddly to iterate
on: every controller-gain change means a battery cycle and a stopwatch. This
project simulates the entire stack -- track, camera, vision pipeline, robot
dynamics -- so that computer-vision and control-systems engineering can be
iterated on and benchmarked with the same rigor as any other software
system: reproducible seeds, automated tests on synthetic imagery, and a
quantitative comparison across controllers, tracks and noise levels instead
of "it looked fine on the bench."

## Architecture

```mermaid
flowchart LR
    subgraph Track["Track generation"]
        TG["generator.py\nperturbed control points\n-> periodic B-spline\n-> arc-length resample"]
        TR["rendering.py\nfloor texture, lighting,\nocclusions, line, crossings"]
        TG --> TR
    end

    subgraph Sim["Closed-loop simulation (sim/world.py)"]
        CAM["camera.py\npinhole model\nhomography render"]
        ROBOT["robot.py\ndifferential drive\nmotor lag + noise"]
    end

    subgraph Vision["Vision pipeline (vision/)"]
        BEV["birdseye.py\nIPM homography"]
        TH["threshold.py\nadaptive threshold"]
        SW["sliding_window.py\nline-pixel search"]
        FIT["fit.py\npolynomial fit"]
        BEV --> TH --> SW --> FIT
    end

    subgraph Control["control/"]
        PID["PID"]
        PP["Pure Pursuit"]
        ST["Stanley"]
    end

    TR -- "world raster" --> CAM
    CAM -- "camera frame" --> BEV
    FIT -- "lateral error,\nheading error,\ncurvature" --> PID
    FIT --> PP
    FIT --> ST
    PID -- "omega" --> ROBOT
    PP -- "omega" --> ROBOT
    ST -- "omega" --> ROBOT
    ROBOT -- "true pose" --> CAM
    ROBOT -- "cross-track error\n(vs. track ground truth)" --> METRICS["benchmark/\nmetrics + plots"]
```

Every frame: the camera renders what the robot sees from its true pose, the
vision pipeline turns that frame into `(lateral_error, heading_error,
curvature)`, an EMA filter smooths it, a curvature-scheduled target speed is
picked, the active controller turns those three numbers into an angular
velocity, and the robot's motor/kinematic model integrates the new pose.
Cross-track error for benchmarking is measured against the track's own
ground-truth centerline, **not** the vision system's self-reported error.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run one closed-loop simulation and print a summary
python scripts/run_demo.py --track oval --controller pid

# Run the full benchmark (all tracks x controllers x noise levels)
python scripts/run_benchmark.py

# Export an annotated GIF of one run
python scripts/make_gif.py --track curvy_loop --controller stanley

# Tests, lint, types
pytest
ruff check .
ruff format --check .
mypy src
```

Requires Python 3.11+. Dependencies: NumPy, OpenCV (headless), SciPy,
Matplotlib, Pillow -- see `pyproject.toml`.

## Usage

| Script | What it does |
| --- | --- |
| `scripts/run_demo.py` | Runs one closed-loop simulation (`--track`, `--controller`, `--seed`) and prints a metrics summary. |
| `scripts/run_benchmark.py` | Runs the full sweep (3 tracks x 3 controllers x 3 noise levels x N seeds), writes `docs/benchmark_results.csv`, `docs/benchmark_summary.csv` and the comparison figures in `docs/img/`. |
| `scripts/make_gif.py` | Runs one simulation and exports an annotated GIF (top-down trajectory + camera inset) to `docs/img/`. |

The library itself is importable as `vision_line_follower` (src layout); see
the package docstrings (`vision_line_follower/__init__.py` and each
subpackage) for the public API.

## The vision pipeline

1. **Bird's-eye warp (IPM)** -- the forward camera view is perspective-warped
   into a top-down patch of the ground ahead of the robot, using a homography
   derived analytically from the camera's own intrinsic/extrinsic
   calibration (not fitted from point correspondences).
2. **Adaptive threshold** -- `cv2.adaptiveThreshold` isolates line pixels
   robustly under the lighting gradients and vignetting the renderer injects.
3. **Sliding-window search** -- starting from a histogram peak near the
   robot, stacked windows walk forward, recentring on detected pixels; an
   optional prior-position hint keeps the tracker anchored to the correct
   branch of the line through an X-crossing.
4. **Polynomial fit** -- a quadratic `lateral = f(forward)` fit yields lateral
   error, heading error (`atan` of the local slope) and curvature
   (`f'' / (1 + f'^2)^1.5`) in one shot.

See [docs/LEARNING.md](docs/LEARNING.md) for the full derivation of the
camera homography and the vision pipeline's math.

## The three controllers

All three consume the *same* three scalars from the vision pipeline and share
a curvature-scheduled speed profile, so the benchmark isolates the steering
law itself.

| Controller | Core idea | Adapted for differential drive by |
| --- | --- | --- |
| **PID** | Drive a blended `lateral_error + heading_weight * heading_error` to zero. | Directly outputs `omega` (no adaptation needed). |
| **Pure Pursuit** | Fit a circular arc to a lookahead point on the path. | Reconstructing a local quadratic path model (Taylor expansion) from the pipeline's three scalars, since no full path is available. |
| **Stanley** | Bicycle-model steering angle from heading + cross-track error. | Mapping the steering angle to `omega` via `omega = v*tan(delta)/L_eff`. |

See `docs/LEARNING.md` for the full equations and the design rationale
behind each adaptation.

## Benchmark results

All numbers below come from actually running `python scripts/run_benchmark.py`
in this repository: 3 tracks x 3 controllers x 3 noise levels x 4 seeds = 108
closed-loop simulations, 596 s total (5.5 s/run average). Noise levels bundle
lighting-gradient strength, occlusion density, floor-texture noise, camera
pixel noise and wheel-speed process noise into three presets (`clean`,
`moderate`, `harsh`) -- see `NoiseLevel` in
`src/vision_line_follower/benchmark/run_benchmark.py`. Raw per-run data:
[`docs/benchmark_results.csv`](docs/benchmark_results.csv); aggregated:
[`docs/benchmark_summary.csv`](docs/benchmark_summary.csv).

**By controller x noise level** (mean over 3 tracks x 4 seeds = 12 runs per cell):

| Controller | Noise | Success rate | Mean \|error\| | Max \|error\| | Mean speed |
| --- | --- | ---: | ---: | ---: | ---: |
| PID | clean | 100.0% | 1.79 cm | 7.18 cm | 0.255 m/s |
| PID | moderate | 100.0% | 1.81 cm | 7.51 cm | 0.251 m/s |
| PID | harsh | 83.3% | 1.91 cm | 8.24 cm | 0.246 m/s |
| Pure Pursuit | clean | 66.7% | 1.11 cm | 2.94 cm | 0.240 m/s |
| Pure Pursuit | moderate | 66.7% | 1.13 cm | 3.28 cm | 0.239 m/s |
| Pure Pursuit | harsh | 58.3% | 1.41 cm | 9.17 cm | 0.235 m/s |
| Stanley | clean | 100.0% | 1.68 cm | 3.70 cm | 0.261 m/s |
| Stanley | moderate | 100.0% | 1.68 cm | 4.59 cm | 0.259 m/s |
| Stanley | harsh | 100.0% | 1.75 cm | 5.03 cm | 0.258 m/s |

**By controller x track** (mean over 3 noise levels x 4 seeds = 12 runs per cell):

| Controller | Track | Success rate | Mean \|error\| | Max \|error\| |
| --- | --- | ---: | ---: | ---: |
| PID | oval | 100.0% | 1.67 cm | 7.39 cm |
| PID | figure_eight | 83.3% | 1.94 cm | 8.24 cm |
| PID | curvy_loop | 100.0% | 1.89 cm | 6.32 cm |
| Pure Pursuit | oval | 100.0% | 1.31 cm | 4.41 cm |
| Pure Pursuit | figure_eight | 0.0% | 1.13 cm | 9.17 cm |
| Pure Pursuit | curvy_loop | 91.7% | 1.22 cm | 7.07 cm |
| Stanley | oval | 100.0% | 1.81 cm | 5.03 cm |
| Stanley | figure_eight | 100.0% | 1.65 cm | 4.29 cm |
| Stanley | curvy_loop | 100.0% | 1.65 cm | 4.78 cm |

**Takeaways.** Stanley is the only controller with a 100% success rate on
every track x noise combination and the most consistent error spread (see
the boxplot below). PID is a close second, with its only dip on `harsh`
noise on the figure-eight track. Pure Pursuit has the *lowest* mean error
when it succeeds (its geometric arc-fitting is precise on tracks it can
track), but it **never** completes the figure-eight track's self-crossing in
this benchmark (0/12) -- a genuine, reproducible weakness, not a fluke or a
bug: it comes from Pure Pursuit's commandable curvature being geometrically
capped by `2 / lookahead`, colliding with the low scheduled speed at that
track's tightest curvature. A curvature-adaptive lookahead reduction helps at
that crossing but breaks the loop-closing maneuver on the other two tracks,
so it was deliberately **not** adopted as the default -- see
`docs/LEARNING.md` section 4 for the full investigation and rejected fix.

Figures (regenerate with `python scripts/run_benchmark.py`):

![Mean cross-track error by controller and noise level](docs/img/benchmark_mean_error.png)
![Success rate by controller and noise level](docs/img/benchmark_success_rate.png)
![Error distribution across all benchmark runs](docs/img/benchmark_error_boxplot.png)

## Project structure

```
src/vision_line_follower/
├── geometry.py         # Pose, wrap_angle, nearest-point-on-polyline
├── track/               # generator.py (spline tracks), rendering.py (world raster)
├── sim/                 # camera.py (pinhole/homography), robot.py (diff-drive), world.py (closed loop)
├── vision/               # birdseye.py, threshold.py, sliding_window.py, fit.py, pipeline.py, temporal_filter.py
├── control/              # base.py, pid.py, pure_pursuit.py, stanley.py
├── benchmark/            # run_benchmark.py (sweep + CSV/aggregate)
└── viz/                  # plots.py (matplotlib figures), gif_export.py

scripts/                 # run_demo.py, run_benchmark.py, make_gif.py
tests/                    # pytest suite (synthetic-image vision tests, kinematics, controllers, integration)
docs/                     # LEARNING.md, benchmark CSVs, img/
```

## Testing & CI

`pytest` covers: track-generation invariants (closed-loop heading continuity,
deterministic seeding, curvature bounds), the vision pipeline against
synthetic bird's-eye images (straight/angled lines, prior-hint branch
tracking) and against the real camera+track stack, robot kinematics
(straight-line motion, in-place rotation, saturation, motor-lag
convergence), controller sign/saturation behavior, and short closed-loop
integration runs. GitHub Actions runs `ruff check`, `ruff format --check`,
`mypy` and `pytest` on Python 3.11 and 3.12 for every push/PR.

## Known limitations & future work

- **Pure Pursuit at the figure-eight self-crossing** (one benchmark seed):
  documented in detail in `docs/LEARNING.md` section 4 -- a genuine,
  known-in-the-literature weakness of Pure Pursuit under very tight curvature
  combined with low speed, not a hidden bug.
- No lens distortion model (ideal pinhole only).
- No tire/ground contact dynamics or lateral slip -- kinematic integration
  with first-order motor lag only.
- No dedicated intersection-handling logic (crossings are geometric/visual
  only; no controller decides "which branch to take").
- Synthetic noise/texture/lighting, not a real-image dataset.

## License

[MIT](LICENSE)
