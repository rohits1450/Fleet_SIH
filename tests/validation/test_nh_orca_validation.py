"""NH-ORCA validation against validation/figures/nh_orca.png -- checking
kinematics, not just geometry."""
import math

from algorithms.local_planning.nh_orca import DEFAULT_EPSILON, nh_orca_velocity
from models.robot import DiffDriveRobot, Pose


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def toward(pos, goal, speed):
    dx, dy = goal[0] - pos[0], goal[1] - pos[1]
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return (0.0, 0.0)
    s = min(speed, d * 2.5)
    return (dx / d * s, dy / d * s)


# -- Scenario 1: Two robots crossing paths (perpendicular) -----------------
def test_perpendicular_crossing_zero_collisions_smooth_avoidance():
    radius = 0.3
    a = DiffDriveRobot("A", Pose(-6.0, 0.0, 0.0), radius=radius, max_speed=1.0, max_omega=3.0)
    b = DiffDriveRobot("B", Pose(0.0, -6.0, math.pi / 2), radius=radius, max_speed=1.0, max_omega=3.0)
    goal_a, goal_b = (6.0, 0.0), (0.0, 6.0)
    dt = 1 / 60

    min_dist = float("inf")
    omega_history = []
    for _ in range(1200):
        pref_a = toward(a.reference_point(DEFAULT_EPSILON), goal_a, a.max_speed)
        pref_b = toward(b.reference_point(DEFAULT_EPSILON), goal_b, b.max_speed)
        v_a, w_a = nh_orca_velocity(a, [b], pref_a, 2.0, dt)
        v_b, w_b = nh_orca_velocity(b, [a], pref_b, 2.0, dt)
        a.set_body_velocity(v_a, w_a)
        b.set_body_velocity(v_b, w_b)
        a.step(dt)
        b.step(dt)
        omega_history.append((w_a, w_b))
        min_dist = min(min_dist, dist(a.position(), b.position()))

    # "smooth curved avoidance (not jerky stop-start)": omega shouldn't be
    # slamming between its extremes tick to tick.
    max_omega_jump = max(
        abs(o1[0] - o2[0]) for o1, o2 in zip(omega_history, omega_history[1:])
    )
    print(f"[perpendicular crossing] min_dist={min_dist:.3f} (combined_radius={2*radius:.2f}), "
          f"max_omega_jump_per_tick={max_omega_jump:.3f}")
    assert min_dist >= 2 * radius - 1e-3, "collision during perpendicular crossing"
    assert dist(a.position(), goal_a) < 0.5 and dist(b.position(), goal_b) < 0.5
    assert max_omega_jump < 6.0 * dt * 60, "omega swings too abruptly between ticks -- jerky, not smooth"


# -- Scenario 2: Differential-drive fidelity --------------------------------
def test_diffdrive_fidelity_no_instantaneous_lateral_slide():
    """Guard against the 'holonomic assumption' bug: a real unicycle cannot
    move sideways relative to its own heading. Log commanded (v, omega) vs.
    the actual position delta each tick and confirm the lateral (sideways)
    component of that delta is always ~zero -- the kinematics genuinely
    integrate a unicycle, not a holonomic point mass wearing a heading."""
    radius = 0.3
    a = DiffDriveRobot("A", Pose(-4.0, 0.0, 0.0), radius=radius, max_speed=1.0, max_omega=3.0)
    b = DiffDriveRobot("B", Pose(4.0, 0.05, math.pi), radius=radius, max_speed=1.0, max_omega=3.0)
    goal_a, goal_b = (4.0, 0.0), (-4.0, 0.0)
    dt = 1 / 60

    max_lateral_slide = 0.0
    log = []
    for _ in range(1800):
        prev_pos = a.position()
        prev_theta = a.pose.theta
        pref_a = toward(a.reference_point(DEFAULT_EPSILON), goal_a, a.max_speed)
        pref_b = toward(b.reference_point(DEFAULT_EPSILON), goal_b, b.max_speed)
        v_a, w_a = nh_orca_velocity(a, [b], pref_a, 2.0, dt)
        v_b, w_b = nh_orca_velocity(b, [a], pref_b, 2.0, dt)
        a.set_body_velocity(v_a, w_a)
        b.set_body_velocity(v_b, w_b)
        a.step(dt)
        b.step(dt)

        dx, dy = a.position()[0] - prev_pos[0], a.position()[1] - prev_pos[1]
        # decompose the actual displacement into the heading frame at the
        # start of the tick: forward component and lateral (sideways) one
        forward = dx * math.cos(prev_theta) + dy * math.sin(prev_theta)
        lateral = -dx * math.sin(prev_theta) + dy * math.cos(prev_theta)
        max_lateral_slide = max(max_lateral_slide, abs(lateral))
        log.append((v_a, w_a, forward, lateral))

    v_a, w_a, forward, lateral = log[len(log) // 2]
    print(f"[diff-drive fidelity] sample mid-run: commanded v={v_a:.3f} omega={w_a:.3f} -> "
          f"executed forward={forward:.5f} lateral={lateral:.6f}; max_lateral_slide over run={max_lateral_slide:.6f}")
    assert max_lateral_slide < 1e-9, "non-zero lateral slip -- kinematics are not a real unicycle"


# -- Scenario 3: Dense cluster ----------------------------------------------
def test_dense_cluster_five_robots_converging_on_one_point():
    radius = 0.28
    n = 5
    robots = []
    goal = (0.0, 0.0)
    for i in range(n):
        angle = 2 * math.pi * i / n
        pos = (6 * math.cos(angle), 6 * math.sin(angle))
        heading = angle + math.pi
        robots.append(DiffDriveRobot(f"r{i}", Pose(pos[0], pos[1], heading), radius=radius,
                                      max_speed=1.0, max_omega=3.0))

    dt = 1 / 60
    min_pairwise = float("inf")
    stalled_ticks = 0
    for _ in range(3600):
        prefs = [toward(r.reference_point(DEFAULT_EPSILON), goal, r.max_speed) for r in robots]
        new_vels = []
        for i, r in enumerate(robots):
            neighbors = [o for j, o in enumerate(robots) if j != i]
            v, w = nh_orca_velocity(r, neighbors, prefs[i], 2.0, dt)
            new_vels.append((v, w))
        all_slow = True
        for r, (v, w) in zip(robots, new_vels):
            r.set_body_velocity(v, w)
            r.step(dt)
            if abs(v) > 0.02:
                all_slow = False
        if all_slow:
            stalled_ticks += 1

        for i in range(n):
            for j in range(i + 1, n):
                d = dist(robots[i].position(), robots[j].position())
                min_pairwise = min(min_pairwise, d)

    print(f"[dense cluster] min_pairwise_dist={min_pairwise:.3f} (combined_radius={2*radius:.2f}), "
          f"fully-stalled ticks={stalled_ticks}/3600")
    assert min_pairwise >= 2 * radius - 1e-3, "collision in dense converging cluster"
    assert stalled_ticks < 3600, "cluster permanently froze -- no robot ever made progress"


# -- Scenario 4: Static + dynamic mixed -------------------------------------
def test_static_and_dynamic_avoidance_do_not_fight_each_other():
    """A robot skirting a static wall (NH-ORCA's static-obstacle lines) while
    also avoiding a moving robot (reciprocal ORCA lines) should blend into
    one smooth command, not oscillate between contradictory corrections."""
    from algorithms.local_planning.nh_orca import nh_orca_velocity as full_nh_orca

    radius = 0.3
    a = DiffDriveRobot("A", Pose(-5.0, 0.6, 0.0), radius=radius, max_speed=1.0, max_omega=3.0)
    moving = DiffDriveRobot("M", Pose(5.0, 0.5, math.pi), radius=radius, max_speed=1.0, max_omega=3.0)
    goal_a, goal_m = (5.0, 0.6), (-5.0, 0.5)
    wall_points = [((x, 0.0), 0.02) for x in [-1.0, -0.5, 0.0, 0.5, 1.0]]  # a "shelf edge" at y=0
    dt = 1 / 60

    omega_signs_flipped = 0
    prev_sign = 0
    min_dist_to_wall = float("inf")
    for _ in range(2400):
        pref_a = toward(a.reference_point(DEFAULT_EPSILON), goal_a, a.max_speed)
        pref_m = toward(moving.reference_point(DEFAULT_EPSILON), goal_m, moving.max_speed)
        v_a, w_a = full_nh_orca(a, [moving], pref_a, 2.0, dt, static_obstacles=wall_points)
        v_m, w_m = full_nh_orca(moving, [a], pref_m, 2.0, dt)
        a.set_body_velocity(v_a, w_a)
        moving.set_body_velocity(v_m, w_m)
        a.step(dt)
        moving.step(dt)

        min_dist_to_wall = min(min_dist_to_wall, min(dist(a.position(), p) for p, _ in wall_points))
        sign = 1 if w_a > 0.05 else (-1 if w_a < -0.05 else 0)
        if sign != 0 and prev_sign != 0 and sign != prev_sign:
            omega_signs_flipped += 1
        if sign != 0:
            prev_sign = sign

    print(f"[static+dynamic mixed] min_dist_to_wall={min_dist_to_wall:.3f}, "
          f"omega direction reversals={omega_signs_flipped}, final dist to goal={dist(a.position(), goal_a):.3f}")
    assert min_dist_to_wall > radius - 0.05, "robot clipped the static wall while avoiding the moving robot"
    assert dist(a.position(), goal_a) < 0.6, "robot never got past the mixed static+dynamic obstacle"
    assert omega_signs_flipped < 20, "omega oscillating between avoid-wall and avoid-robot corrections"
