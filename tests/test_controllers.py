import pytest

from vision_line_follower.control.pid import PIDConfig, PIDController
from vision_line_follower.control.pure_pursuit import PurePursuitConfig, PurePursuitController
from vision_line_follower.control.stanley import StanleyConfig, StanleyController


class TestPIDController:
    def test_zero_error_zero_omega(self) -> None:
        controller = PIDController(PIDConfig())
        omega = controller.compute(0.0, 0.0, 0.0, speed_mps=0.3, dt=0.02)
        assert omega == pytest.approx(0.0)

    def test_positive_lateral_error_turns_left(self) -> None:
        controller = PIDController(PIDConfig())
        omega = controller.compute(0.05, 0.0, 0.0, speed_mps=0.3, dt=0.02)
        assert omega > 0

    def test_negative_lateral_error_turns_right(self) -> None:
        controller = PIDController(PIDConfig())
        omega = controller.compute(-0.05, 0.0, 0.0, speed_mps=0.3, dt=0.02)
        assert omega < 0

    def test_reset_clears_integral_state(self) -> None:
        controller = PIDController(PIDConfig())
        for _ in range(20):
            controller.compute(0.05, 0.0, 0.0, speed_mps=0.3, dt=0.02)
        assert controller._integral != 0.0
        controller.reset()
        assert controller._integral == 0.0
        assert controller._prev_error is None

    def test_output_is_saturated(self) -> None:
        controller = PIDController(PIDConfig(omega_limit=1.0, kp=100.0, ki=0.0, kd=0.0))
        omega = controller.compute(1.0, 0.0, 0.0, speed_mps=0.3, dt=0.02)
        assert abs(omega) <= 1.0 + 1e-9


class TestPurePursuitController:
    def test_zero_error_zero_omega(self) -> None:
        controller = PurePursuitController(PurePursuitConfig())
        omega = controller.compute(0.0, 0.0, 0.0, speed_mps=0.3, dt=0.02)
        assert omega == pytest.approx(0.0, abs=1e-9)

    def test_positive_lateral_error_turns_left(self) -> None:
        controller = PurePursuitController(PurePursuitConfig())
        omega = controller.compute(0.05, 0.0, 0.0, speed_mps=0.3, dt=0.02)
        assert omega > 0

    def test_output_is_saturated(self) -> None:
        controller = PurePursuitController(PurePursuitConfig(omega_limit=0.5))
        omega = controller.compute(5.0, 0.0, 0.0, speed_mps=1.0, dt=0.02)
        assert abs(omega) <= 0.5 + 1e-9


class TestStanleyController:
    def test_zero_error_zero_omega(self) -> None:
        controller = StanleyController(StanleyConfig())
        omega = controller.compute(0.0, 0.0, 0.0, speed_mps=0.3, dt=0.02)
        assert omega == pytest.approx(0.0, abs=1e-9)

    def test_positive_lateral_error_turns_left(self) -> None:
        controller = StanleyController(StanleyConfig())
        omega = controller.compute(0.05, 0.0, 0.0, speed_mps=0.3, dt=0.02)
        assert omega > 0

    def test_heading_error_dominates_at_zero_lateral_error(self) -> None:
        controller = StanleyController(StanleyConfig())
        omega = controller.compute(0.0, 0.3, 0.0, speed_mps=0.3, dt=0.02)
        assert omega > 0

    def test_output_is_saturated(self) -> None:
        controller = StanleyController(StanleyConfig(omega_limit=0.5))
        omega = controller.compute(5.0, 0.0, 0.0, speed_mps=1.0, dt=0.02)
        assert abs(omega) <= 0.5 + 1e-9
