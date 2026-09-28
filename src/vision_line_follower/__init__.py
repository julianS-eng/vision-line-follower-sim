"""Vision Line Follower: a simulated differential-drive robot that follows a
painted line using a classical computer-vision pipeline.

The package is organised by concern:

- :mod:`vision_line_follower.track` -- procedural 2D track generation and rendering.
- :mod:`vision_line_follower.sim` -- robot kinematics/dynamics and camera simulation.
- :mod:`vision_line_follower.vision` -- the OpenCV perception pipeline.
- :mod:`vision_line_follower.control` -- PID, Pure Pursuit and Stanley controllers.
- :mod:`vision_line_follower.benchmark` -- multi-track, multi-controller evaluation.
- :mod:`vision_line_follower.viz` -- plotting and GIF export helpers.
"""

__version__ = "0.1.0"
