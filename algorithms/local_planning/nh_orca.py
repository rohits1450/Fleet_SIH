"""NH-ORCA: apply ORCA at a reference point offset epsilon ahead of a
differential-drive robot's wheel axle, so the holonomic velocity-obstacle
math is valid for a non-holonomic unicycle (Alonso-Mora et al., "Optimal
Reciprocal Collision Avoidance for Multiple Non-Holonomic Robots").

Two adaptations on top of plain ORCA:
  1. Reference-point shift: compute everything at p + eps*heading instead
     of the wheel-axle center, since a unicycle can't instantaneously
     realize an arbitrary holonomic velocity at its own center.
  2. Minkowski-sum radius enlargement: the true robot disk, viewed from the
     reference point, wobbles by up to eps as heading changes, so inflate
     the effective radius by eps to keep the disk contained.
"""
from __future__ import annotations

from algorithms.local_planning.orca import Line, compute_orca_line, solve_velocity
from models.robot import DiffDriveRobot

DEFAULT_EPSILON = 0.18


def nh_orca_velocity(
    robot: DiffDriveRobot,
    neighbors: list[DiffDriveRobot],
    preferred_reference_velocity: tuple[float, float],
    time_horizon: float,
    time_step: float,
    epsilon: float = DEFAULT_EPSILON,
    static_obstacles: list[tuple[tuple[float, float], float]] | None = None,
    static_time_horizon: float | None = None,
) -> tuple[float, float]:
    """Returns (v, omega) for `robot` that avoids `neighbors` under NH-ORCA.
    `preferred_reference_velocity` is the desired (vx, vy) at the robot's
    epsilon-offset reference point (e.g. toward its next waypoint).

    `static_obstacles` is an optional list of (point, radius) pairs -- e.g.
    the nearest point on a nearby wall/pod -- each treated as a zero-velocity
    agent this robot takes full (not reciprocal) responsibility for avoiding,
    since the wall won't move to help. Without this, NH-ORCA only reasons
    about other robots and a maneuver to dodge one can walk straight into
    static geometry it doesn't know exists.

    `static_time_horizon` (defaults to `time_horizon` if unset) lets static
    lines use a shorter horizon than reciprocal robot-robot ones. ORCA's
    velocity obstacle is a *linear* extrapolation of current velocity: a
    long horizon is right for another robot (whose true motion you want to
    react to early) but overstates a static corner's threat when the
    robot's actual path curves around it -- worse the larger (and slower)
    the robot's own footprint, since the extrapolated cone scales with it.
    That's what stalls a big, slow robot right at a row transition next to
    a wall corner that a smaller/faster one glides past without issue."""
    ref_pos = robot.reference_point(epsilon)
    ref_vel = robot.velocity_at_reference_point(epsilon)
    eff_radius = robot.radius + epsilon
    obstacle_horizon = static_time_horizon if static_time_horizon is not None else time_horizon

    lines: list[Line] = []
    for other in neighbors:
        if other is robot:
            continue
        other_ref_pos = other.reference_point(epsilon)
        other_ref_vel = other.velocity_at_reference_point(epsilon)
        other_eff_radius = other.radius + epsilon
        lines.append(
            compute_orca_line(
                ref_pos, ref_vel, eff_radius,
                other_ref_pos, other_ref_vel, other_eff_radius,
                time_horizon, time_step,
            )
        )

    for obstacle_point, obstacle_radius in static_obstacles or []:
        lines.append(
            compute_orca_line(
                ref_pos, ref_vel, eff_radius,
                obstacle_point, (0.0, 0.0), obstacle_radius,
                obstacle_horizon, time_step,
                self_responsibility=1.0,
            )
        )

    max_ref_speed = robot.max_speed + epsilon * robot.max_omega
    new_ref_vel = solve_velocity(lines, max_ref_speed, preferred_reference_velocity)
    return robot.body_velocity_from_reference_velocity(new_ref_vel[0], new_ref_vel[1], epsilon)
