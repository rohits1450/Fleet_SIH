"""ORCA (Optimal Reciprocal Collision Avoidance), van den Berg et al. 2011.

Holonomic core: given a preferred velocity and a set of nearby (position,
velocity, radius) agents, builds one linear half-plane constraint per
neighbor and solves the small 2D linear program for the feasible velocity
closest to preferred. This mirrors the RVO2 reference math (Agent::
computeNewVelocity / linearProgram1-3) so it inherits its correctness and
its graceful degradation (linearProgram3) when constraints are infeasible.

The non-holonomic reference-point adaptation lives in nh_orca.py; this
module only knows about points, velocities, and radii.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

Vec2 = tuple[float, float]


def dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def det(a: Vec2, b: Vec2) -> float:
    return a[0] * b[1] - a[1] * b[0]


def sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def scale(a: Vec2, s: float) -> Vec2:
    return (a[0] * s, a[1] * s)


def norm(a: Vec2) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1])


def norm_sq(a: Vec2) -> float:
    return a[0] * a[0] + a[1] * a[1]


def normalized(a: Vec2) -> Vec2:
    n = norm(a)
    if n < 1e-12:
        return (0.0, 0.0)
    return (a[0] / n, a[1] / n)


@dataclass
class Line:
    point: Vec2
    direction: Vec2  # unit vector; feasible half-plane is where det(direction, v - point) <= 0


def compute_orca_line(
    position: Vec2,
    velocity: Vec2,
    radius: float,
    other_position: Vec2,
    other_velocity: Vec2,
    other_radius: float,
    time_horizon: float,
    time_step: float,
    self_responsibility: float = 0.5,
) -> Line:
    """One ORCA half-plane constraining THIS agent's next velocity.

    `self_responsibility` is the fraction of the avoidance velocity change
    this agent takes on: 0.5 for reciprocal agent-agent avoidance (both
    sides give way equally), 1.0 for a static obstacle (it won't move, so
    this agent must do all of the avoiding)."""
    relative_position = sub(other_position, position)
    relative_velocity = sub(velocity, other_velocity)
    dist_sq = norm_sq(relative_position)
    combined_radius = radius + other_radius
    combined_radius_sq = combined_radius * combined_radius

    if dist_sq > combined_radius_sq:
        inv_time_horizon = 1.0 / time_horizon
        w = sub(relative_velocity, scale(relative_position, inv_time_horizon))
        w_length_sq = norm_sq(w)
        dot_product1 = dot(w, relative_position)

        if dot_product1 < 0.0 and dot_product1 * dot_product1 > combined_radius_sq * w_length_sq:
            w_length = math.sqrt(w_length_sq)
            unit_w = scale(w, 1.0 / w_length)
            direction = (unit_w[1], -unit_w[0])
            u = scale(unit_w, combined_radius * inv_time_horizon - w_length)
        else:
            leg = math.sqrt(max(0.0, dist_sq - combined_radius_sq))
            if det(relative_position, w) > 0.0:
                direction = (
                    (relative_position[0] * leg - relative_position[1] * combined_radius) / dist_sq,
                    (relative_position[0] * combined_radius + relative_position[1] * leg) / dist_sq,
                )
            else:
                direction = (
                    -(relative_position[0] * leg + relative_position[1] * combined_radius) / dist_sq,
                    -(-relative_position[0] * combined_radius + relative_position[1] * leg) / dist_sq,
                )
            dot_product2 = dot(relative_velocity, direction)
            u = sub(scale(direction, dot_product2), relative_velocity)
    else:
        # Already overlapping: fall back to a same-time-step cutoff circle
        # so agents actively separate instead of freezing.
        inv_time_step = 1.0 / time_step
        w = sub(relative_velocity, scale(relative_position, inv_time_step))
        w_length = norm(w)
        unit_w = scale(w, 1.0 / w_length) if w_length > 1e-12 else (1.0, 0.0)
        direction = (unit_w[1], -unit_w[0])
        u = scale(unit_w, combined_radius * inv_time_step - w_length)

    point = add(velocity, scale(u, self_responsibility))
    return Line(point=point, direction=direction)


def _linear_program_1(
    lines: list[Line], line_no: int, radius: float, opt_velocity: Vec2, direction_opt: bool
) -> tuple[bool, Vec2]:
    """Optimize along lines[line_no] subject to lines[0:line_no] and the
    max-speed disk of the given radius."""
    line = lines[line_no]
    dot_prod = dot(line.point, line.direction)
    discriminant = dot_prod * dot_prod + radius * radius - norm_sq(line.point)
    if discriminant < 0.0:
        return False, (0.0, 0.0)

    sqrt_discriminant = math.sqrt(discriminant)
    t_left = -dot_prod - sqrt_discriminant
    t_right = -dot_prod + sqrt_discriminant

    for i in range(line_no):
        other = lines[i]
        denominator = det(line.direction, other.direction)
        numerator = det(other.direction, sub(line.point, other.point))
        if abs(denominator) <= 1e-12:
            if numerator < 0.0:
                return False, (0.0, 0.0)
            continue
        t = numerator / denominator
        if denominator >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return False, (0.0, 0.0)

    if direction_opt:
        t = t_right if dot(opt_velocity, line.direction) > 0.0 else t_left
    else:
        t = dot(sub(opt_velocity, line.point), line.direction)
        t = max(t_left, min(t_right, t))

    return True, add(line.point, scale(line.direction, t))


def _linear_program_2(
    lines: list[Line], radius: float, opt_velocity: Vec2, direction_opt: bool
) -> tuple[int, Vec2]:
    if direction_opt:
        result = scale(opt_velocity, radius)
    elif norm_sq(opt_velocity) > radius * radius:
        result = scale(normalized(opt_velocity), radius)
    else:
        result = opt_velocity

    for i, line in enumerate(lines):
        if det(line.direction, sub(line.point, result)) > 0.0:
            ok, candidate = _linear_program_1(lines, i, radius, opt_velocity, direction_opt)
            if not ok:
                return i, result
            result = candidate
    return len(lines), result


def _linear_program_3(lines: list[Line], num_start: int, radius: float, result: Vec2) -> Vec2:
    distance = 0.0
    for i in range(num_start, len(lines)):
        line = lines[i]
        if det(line.direction, sub(line.point, result)) > distance:
            proj_lines = []
            for j in range(i):
                other = lines[j]
                d = det(line.direction, other.direction)
                if abs(d) <= 1e-12:
                    if dot(line.direction, other.direction) > 0.0:
                        continue
                    point = scale(add(line.point, other.point), 0.5)
                else:
                    t = det(other.direction, sub(line.point, other.point)) / d
                    point = add(line.point, scale(line.direction, t))
                direction = normalized(sub(other.direction, line.direction))
                proj_lines.append(Line(point=point, direction=direction))

            perp = (-line.direction[1], line.direction[0])
            ok, candidate = _linear_program_2(proj_lines, radius, perp, True)
            if not ok:
                candidate = result  # degenerate: keep current best rather than crash
            result = candidate
            distance = det(line.direction, sub(line.point, result))
    return result


def solve_velocity(
    orca_lines: list[Line], max_speed: float, preferred_velocity: Vec2
) -> Vec2:
    """Feasible velocity inside all ORCA half-planes and the max-speed disk,
    closest to preferred_velocity. Falls back to linearProgram3's
    least-bad-violation solution when the constraints are jointly
    infeasible (dense/deadlocked traffic)."""
    fail_index, result = _linear_program_2(orca_lines, max_speed, preferred_velocity, False)
    if fail_index < len(orca_lines):
        result = _linear_program_3(orca_lines, fail_index, max_speed, result)
    return result
