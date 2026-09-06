"""Differential-drive robot: pose, kinematics integration, and a minimal
state machine. No ROS, no Pygame -- reused by every phase."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto


class RobotState(Enum):
    IDLE = auto()
    MOVING = auto()
    AVOIDING = auto()
    DONE = auto()


@dataclass
class Pose:
    x: float
    y: float
    theta: float  # radians


@dataclass
class DiffDriveRobot:
    robot_id: str
    pose: Pose
    radius: float = 0.35
    max_speed: float = 2.0          # m/s, wheel-average linear speed cap
    max_omega: float = 3.0          # rad/s
    wheel_base: float = 0.4         # distance between wheels (m)
    state: RobotState = RobotState.IDLE
    velocity: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))  # (v, omega) applied last step

    def position(self) -> tuple[float, float]:
        return (self.pose.x, self.pose.y)

    def set_wheel_speeds(self, v_left: float, v_right: float) -> None:
        """Convert wheel speeds to body (v, omega), clamped to robot limits."""
        v = (v_left + v_right) / 2.0
        omega = (v_right - v_left) / self.wheel_base
        v = max(-self.max_speed, min(self.max_speed, v))
        omega = max(-self.max_omega, min(self.max_omega, omega))
        self.velocity = (v, omega)

    def set_body_velocity(self, v: float, omega: float) -> None:
        v = max(-self.max_speed, min(self.max_speed, v))
        omega = max(-self.max_omega, min(self.max_omega, omega))
        self.velocity = (v, omega)

    def wheel_speeds_for(self, v: float, omega: float) -> tuple[float, float]:
        v_left = v - omega * self.wheel_base / 2.0
        v_right = v + omega * self.wheel_base / 2.0
        return v_left, v_right

    def step(self, dt: float) -> None:
        """Integrate unicycle kinematics forward by dt using the last-set
        (v, omega) body velocity."""
        v, omega = self.velocity
        self.pose.x += v * math.cos(self.pose.theta) * dt
        self.pose.y += v * math.sin(self.pose.theta) * dt
        self.pose.theta = (self.pose.theta + omega * dt + math.pi) % (2 * math.pi) - math.pi

    def reference_point(self, epsilon: float) -> tuple[float, float]:
        """Point offset epsilon ahead of the wheel axle center along heading.
        NH-ORCA treats this point as a holonomic agent; epsilon absorbs the
        non-holonomic tracking error so ORCA's linear velocity constraints
        remain valid for a unicycle."""
        return (
            self.pose.x + epsilon * math.cos(self.pose.theta),
            self.pose.y + epsilon * math.sin(self.pose.theta),
        )

    def velocity_at_reference_point(self, epsilon: float) -> tuple[float, float]:
        v, omega = self.velocity
        vx = v * math.cos(self.pose.theta) - epsilon * omega * math.sin(self.pose.theta)
        vy = v * math.sin(self.pose.theta) + epsilon * omega * math.cos(self.pose.theta)
        return (vx, vy)

    def body_velocity_from_reference_velocity(
        self, vx_ref: float, vy_ref: float, epsilon: float
    ) -> tuple[float, float]:
        """Invert the reference-point Jacobian to recover (v, omega) that
        realizes a desired holonomic velocity at the epsilon-offset point."""
        c, s = math.cos(self.pose.theta), math.sin(self.pose.theta)
        v = vx_ref * c + vy_ref * s
        omega = (-vx_ref * s + vy_ref * c) / epsilon
        return v, omega
