"""D* Lite validation against Algovalidations/d* lite.png, tested in
isolation from task allocation and other robots (core.world +
core.planner.dstar_lite only -- no CBBA, no other agents)."""
import math
import time

from core.planner.dstar_lite import DStarLite, astar_expansions
from core.world import World

GRID_W, GRID_H = 30, 20


def make_shelved_world() -> World:
    w = World(GRID_W, GRID_H)
    for row_y in (4, 5, 9, 10, 14, 15):
        for x in range(3, GRID_W - 3):
            if x % 6 in (4, 5):
                continue
            w.add_obstacle((x, row_y))
    return w


def path_cost(world: World, path: list) -> float:
    return sum(world.edge_cost(a, b) for a, b in zip(path, path[1:]))


# -- Scenario 1: Static replan -------------------------------------------
def test_static_replan_is_shortest_available_route():
    world = make_shelved_world()
    start, goal = (1, 1), (GRID_W - 2, GRID_H - 2)
    planner = DStarLite(world, start, goal)
    path = planner.get_path()

    assert path and path[0] == start and path[-1] == goal
    assert all(c not in world.obstacles for c in path)

    optimal_path, _ = astar_expansions(world, start, goal)
    assert abs(path_cost(world, path) - path_cost(world, optimal_path)) < 1e-9, (
        "D* Lite's path is not the shortest available route"
    )
    print(f"[static replan] cost={path_cost(world, path):.3f} matches optimal A* cost")


# -- Scenario 2: Dynamic obstacle mid-path --------------------------------
def test_dynamic_obstacle_dropped_ahead_of_moving_robot_causes_local_detour():
    world = make_shelved_world()
    start, goal = (1, 1), (GRID_W - 2, GRID_H - 2)
    planner = DStarLite(world, start, goal)

    # Drive the robot partway along its own plan.
    for _ in range(8):
        path = planner.get_path()
        assert len(path) > 1, "robot froze before the obstacle was even placed"
        next_cell = path[1]
        planner.update_start(next_cell)
        planner.compute_shortest_path()

    path_before = planner.get_path()
    assert path_before[-1] == goal, "no path before the drop -- test setup invalid"
    ahead_cell = path_before[min(2, len(path_before) - 1)]
    assert ahead_cell != planner.start

    world.add_obstacle(ahead_cell)
    planner.reset_expansion_counter()
    planner.notify_obstacles_changed([ahead_cell])
    path_after = planner.get_path()

    assert path_after, "robot froze (no path) after a single obstacle directly ahead"
    assert path_after[-1] == goal
    assert ahead_cell not in path_after

    # "local detour if one exists": the new path should rejoin the old one
    # quickly rather than diverging into a totally different route. Compare
    # how much of the tail the two paths still share.
    old_tail = set(path_before[3:])
    new_tail = set(path_after[3:]) if len(path_after) > 3 else set()
    shared = old_tail & new_tail
    print(f"[dynamic obstacle] path_before_len={len(path_before)} path_after_len={len(path_after)} "
          f"shared_tail_cells={len(shared)}/{len(old_tail)} expansions_to_repair={planner.node_expansions}")
    assert len(shared) / max(len(old_tail), 1) > 0.5, "detour looks global, not local"


# -- Scenario 3: Node-expansion count -------------------------------------
def test_incremental_repair_expansions_are_small_fraction_of_first_plan():
    world = World(60, 40)
    for row_y in (8, 9, 20, 21, 30, 31):
        for x in range(4, 56):
            if x % 6 in (4, 5):
                continue
            world.add_obstacle((x, row_y))

    start, goal = (1, 1), (58, 38)
    planner = DStarLite(world, start, goal)
    first_plan_expansions = planner.node_expansions
    assert planner.get_path()[-1] == goal

    # Small obstacle far from the bulk of the already-explored frontier.
    obstacle_cell = (30, 15)
    world.add_obstacle(obstacle_cell)
    planner.reset_expansion_counter()
    planner.notify_obstacles_changed([obstacle_cell])
    repair_expansions = planner.node_expansions

    fraction = repair_expansions / first_plan_expansions
    print(f"[node expansions] first_plan={first_plan_expansions} repair={repair_expansions} "
          f"fraction={fraction:.3%}")
    assert fraction < 0.1, "incremental repair touched too large a fraction of the graph"


# -- Scenario 4: Goal becomes unreachable ---------------------------------
def test_goal_walled_off_terminates_immediately_no_infinite_loop():
    world = make_shelved_world()
    start, goal = (1, 1), (10, 6)
    planner = DStarLite(world, start, goal)
    assert planner.get_path()[-1] == goal

    changed = []
    x, y = goal
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        c = (x + dx, y + dy)
        if world.add_obstacle(c):
            changed.append(c)

    t0 = time.monotonic()
    planner.notify_obstacles_changed(changed)
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0, f"compute_shortest_path did not terminate promptly ({elapsed:.2f}s) -- looks like a loop"
    path = planner.get_path()
    assert not path or path[-1] != goal
    assert planner._rhs(planner.start) == float("inf")
    print(f"[unreachable goal] correctly reports no path in {elapsed*1000:.1f}ms, no infinite loop")


def test_fleet_reassigns_task_when_target_becomes_unreachable():
    """Fleet-level half of the same criterion: the robot doesn't just sit
    there -- the task gets handed to another robot instead of infinite
    waiting. (core-level part is test_goal_walled_off above)."""
    import random
    import scenarios.fleet_sim as fs

    random.seed(7)
    sim = fs.FleetSim()
    for _ in range(400):
        sim._step(fs.DT)

    r1 = sim.agents["r1"]
    if r1.state not in (fs.TaskState.TO_PICKUP, fs.TaskState.TO_DROPOFF):
        for _ in range(2000):
            sim._step(fs.DT)
            if r1.state in (fs.TaskState.TO_PICKUP, fs.TaskState.TO_DROPOFF):
                break
    assert r1.task is not None, "test setup failed to get a robot onto a task"
    stuck_task_id = r1.task.task_id

    cell = fs.pos_to_cell(r1.robot.position())
    x, y = cell
    for c in [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]:
        if c not in sim.pod_cells and sim.world.is_free(c):
            sim._handle_click((c[0] * fs.CELL_PX + 5, fs.TOP_BAR + c[1] * fs.CELL_PX + 5))
    assert list(sim.world.neighbors(cell)) == [], "test setup failed to fully box the robot in"

    reassigned = False
    for _ in range(400):
        sim._step(fs.DT)
        if r1.state == fs.TaskState.IDLE:
            reassigned = True
            break

    assert reassigned, "robot never gave up an unreachable task -- looks like infinite waiting"
    # A won task may be queued behind another in the bundle, not necessarily
    # the one actively being driven to -- check the whole bundle/path, not
    # just the single active `.task`.
    took_over = any(
        a is not r1 and any(tid.startswith(stuck_task_id) for tid in a.cbba.bundle)
        for a in sim.agents.values()
    )
    print(f"[fleet reassignment] r1 abandoned '{stuck_task_id}', picked up elsewhere: {took_over}")
    assert took_over, "no other robot picked up the reassigned task"
    # And r1 itself must not keep re-winning its own retries while stuck.
    assert not any(tid.startswith(stuck_task_id) for tid in r1.cbba.bundle), (
        "the stuck robot re-won its own abandoned task"
    )


# -- Scenario 5: Trap-and-release -----------------------------------------
def test_trap_and_release_reconverges_with_no_ghost_cost():
    world = make_shelved_world()
    start, goal = (1, 1), (GRID_W - 2, GRID_H - 2)
    planner = DStarLite(world, start, goal)
    original_path = planner.get_path()
    original_cost = path_cost(world, original_path)

    trap_cell = original_path[len(original_path) // 2]
    world.add_obstacle(trap_cell)
    planner.notify_obstacles_changed([trap_cell])
    detour_path = planner.get_path()
    assert trap_cell not in detour_path
    assert path_cost(world, detour_path) > original_cost - 1e-9

    world.remove_obstacle(trap_cell)
    planner.notify_obstacles_changed([trap_cell])
    released_path = planner.get_path()
    released_cost = path_cost(world, released_path)

    assert abs(released_cost - original_cost) < 1e-9, (
        f"path did not reconverge to original cost: {released_cost} vs {original_cost}"
    )

    # No "ghost" cost lingering: g-values along the reconverged path must
    # match a completely fresh planner built on the identical (now
    # unobstructed) world.
    fresh = DStarLite(world, start, goal)
    for cell in released_path:
        assert math.isclose(planner._g(cell), fresh._g(cell), abs_tol=1e-6), (
            f"stale g-value at {cell}: live={planner._g(cell)} fresh={fresh._g(cell)}"
        )
    print(f"[trap-and-release] reconverged to original cost {original_cost:.3f}, no ghost g-values")
