import math

import numpy as np
import pytest

from vision_line_follower.sim.robot import DifferentialDriveRobot, RobotConfig, RobotState


def test_straight_line_motion() -> None:
    config = RobotConfig(motor_time_constant_s=1e-6, wheel_speed_noise_std=0.0)
    robot = DifferentialDriveRobot(config, RobotState(0.0, 0.0, 0.0))
    for _ in range(50):
        robot.step(0.2, 0.2, dt=0.02)
    assert robot.state.y == pytest.approx(0.0, abs=1e-6)
    assert robot.state.x == pytest.approx(0.2 * 50 * 0.02, rel=1e-2)
    assert robot.state.theta == pytest.approx(0.0, abs=1e-6)


def test_pure_rotation_in_place() -> None:
    config = RobotConfig(motor_time_constant_s=1e-6, wheel_base_m=0.1, wheel_speed_noise_std=0.0)
    robot = DifferentialDriveRobot(config, RobotState(0.0, 0.0, 0.0))
    # Equal and opposite wheel speeds rotate in place.
    for _ in range(100):
        robot.step(-0.1, 0.1, dt=0.02)
    assert robot.state.x == pytest.approx(0.0, abs=1e-6)
    assert robot.state.y == pytest.approx(0.0, abs=1e-6)
    assert robot.state.theta != pytest.approx(0.0, abs=1e-3)


def test_wheel_speed_saturation() -> None:
    config = RobotConfig(max_wheel_speed_mps=0.3, motor_time_constant_s=1e-6)
    robot = DifferentialDriveRobot(config, RobotState(0.0, 0.0, 0.0))
    robot.step(10.0, 10.0, dt=0.02)
    assert abs(robot.state.v_left) <= 0.3 + 1e-9
    assert abs(robot.state.v_right) <= 0.3 + 1e-9


def test_motor_first_order_lag_converges() -> None:
    config = RobotConfig(motor_time_constant_s=0.1, wheel_speed_noise_std=0.0)
    robot = DifferentialDriveRobot(config, RobotState(0.0, 0.0, 0.0))
    for _ in range(500):
        robot.step(0.4, 0.4, dt=0.01)
    assert robot.state.v_left == pytest.approx(0.4, abs=1e-3)
    assert robot.state.v_right == pytest.approx(0.4, abs=1e-3)


def test_motor_lag_does_not_reach_target_instantly() -> None:
    config = RobotConfig(motor_time_constant_s=0.5, wheel_speed_noise_std=0.0)
    robot = DifferentialDriveRobot(config, RobotState(0.0, 0.0, 0.0))
    robot.step(0.5, 0.5, dt=0.01)
    assert 0.0 < robot.state.v_left < 0.1


def test_unicycle_to_wheel_speeds_roundtrip() -> None:
    v, omega = 0.3, 1.5
    wheel_base = 0.14
    v_left, v_right = DifferentialDriveRobot.unicycle_to_wheel_speeds(v, omega, wheel_base)
    recovered_v = (v_left + v_right) / 2.0
    recovered_omega = (v_right - v_left) / wheel_base
    assert recovered_v == pytest.approx(v)
    assert recovered_omega == pytest.approx(omega)


def test_linear_angular_velocity_matches_wheel_speeds() -> None:
    config = RobotConfig(wheel_base_m=0.2)
    robot = DifferentialDriveRobot(config, RobotState(0.0, 0.0, 0.0, v_left=0.1, v_right=0.3))
    v, omega = robot.linear_angular_velocity()
    assert v == pytest.approx(0.2)
    assert omega == pytest.approx(1.0)


def test_process_noise_changes_trajectory() -> None:
    config = RobotConfig(wheel_speed_noise_std=0.05, motor_time_constant_s=0.05)
    robot_a = DifferentialDriveRobot(
        config, RobotState(0.0, 0.0, 0.0), rng=np.random.default_rng(0)
    )
    robot_b = DifferentialDriveRobot(
        config, RobotState(0.0, 0.0, 0.0), rng=np.random.default_rng(1)
    )
    for _ in range(50):
        robot_a.step(0.2, 0.2, dt=0.02)
        robot_b.step(0.2, 0.2, dt=0.02)
    assert not math.isclose(robot_a.state.x, robot_b.state.x, abs_tol=1e-9)
