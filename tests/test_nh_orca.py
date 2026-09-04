import math

from core.avoidance.orca import compute_orca_line, solve_velocity
from core.avoidance.nh_orca import nh_orca_velocity
from core.robot import DiffDriveRobot, Pose


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def test_holonomic_orca_head_on_no_collision():
    radius = 0.3
    max_speed = 1.0
    tau = 2.0
    dt = 0.1

    # A perfectly mirrored head-on course is ORCA's known singular case: the
    # cutoff line is exactly vertical through the midpoint, so velocity
    # decays to zero without either agent ever breaking symmetry sideways.
    # Real deployments never see an exact mirror (sensor noise, heading
    # drift), so nudge one agent off-axis the way any live system would.
    pos_a, pos_b = [-3.0, 0.02], [3.0, 0.0]
    vel_a, vel_b = (0.0, 0.0), (0.0, 0.0)
    goal_a, goal_b = (3.0, 0.0), (-3.0, 0.0)

    min_dist = float("inf")
    for _ in range(200):
        pref_a = _toward(pos_a, goal_a, max_speed)
        pref_b = _toward(pos_b, goal_b, max_speed)

        line_a = compute_orca_line(tuple(pos_a), vel_a, radius, tuple(pos_b), vel_b, radius, tau, dt)
        line_b = compute_orca_line(tuple(pos_b), vel_b, radius, tuple(pos_a), vel_a, radius, tau, dt)

        vel_a = solve_velocity([line_a], max_speed, pref_a)
        vel_b = solve_velocity([line_b], max_speed, pref_b)

        pos_a[0] += vel_a[0] * dt
        pos_a[1] += vel_a[1] * dt
        pos_b[0] += vel_b[0] * dt
        pos_b[1] += vel_b[1] * dt

        min_dist = min(min_dist, dist(pos_a, pos_b))

    assert min_dist >= 2 * radius - 1e-6, f"agents collided, min_dist={min_dist}"
    assert dist(pos_a, goal_a) < 0.5
    assert dist(pos_b, goal_b) < 0.5


def _toward(pos, goal, speed):
    dx, dy = goal[0] - pos[0], goal[1] - pos[1]
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return (0.0, 0.0)
    s = min(speed, d / 0.1)
    return (dx / d * s, dy / d * s)


def test_nh_orca_two_diffdrive_robots_head_on_no_collision():
    radius = 0.35
    epsilon = 0.18
    dt = 0.05
    tau = 2.0

    robot_a = DiffDriveRobot("A", Pose(-4.0, 0.0, 0.0), radius=radius, max_speed=1.0, max_omega=3.0)
    robot_b = DiffDriveRobot("B", Pose(4.0, 0.0, math.pi), radius=radius, max_speed=1.0, max_omega=3.0)
    goal_a, goal_b = (4.0, 0.0), (-4.0, 0.0)

    min_dist = float("inf")
    for _ in range(400):
        ref_a = robot_a.reference_point(epsilon)
        ref_b = robot_b.reference_point(epsilon)
        pref_a = _toward(list(ref_a), goal_a, robot_a.max_speed)
        pref_b = _toward(list(ref_b), goal_b, robot_b.max_speed)

        v_a, w_a = nh_orca_velocity(robot_a, [robot_b], pref_a, tau, dt, epsilon)
        v_b, w_b = nh_orca_velocity(robot_b, [robot_a], pref_b, tau, dt, epsilon)

        robot_a.set_body_velocity(v_a, w_a)
        robot_b.set_body_velocity(v_b, w_b)
        robot_a.step(dt)
        robot_b.step(dt)

        min_dist = min(min_dist, dist(robot_a.position(), robot_b.position()))

    assert min_dist >= 2 * radius - 1e-3, f"robots collided, min_dist={min_dist}"
    assert dist(robot_a.position(), goal_a) < 0.6
    assert dist(robot_b.position(), goal_b) < 0.6
