from environment.grid_world import World
from algorithms.global_planning.dstar_lite import DStarLite, astar_expansions


def test_straight_line_no_obstacles():
    w = World(10, 10)
    d = DStarLite(w, (0, 0), (9, 0))
    path = d.get_path()
    assert path[0] == (0, 0)
    assert path[-1] == (9, 0)
    assert len(path) == 10


def test_routes_around_wall():
    w = World(10, 10)
    for y in range(0, 8):
        w.add_obstacle((5, y))
    d = DStarLite(w, (0, 0), (9, 0))
    path = d.get_path()
    assert path[-1] == (9, 0)
    assert all(c not in w.obstacles for c in path)


def test_local_obstacle_replans_far_fewer_nodes_than_full_astar():
    """The scenario D* Lite is actually built for: a small obstacle dropped
    in an already-explored map, away from the bulk of the optimal path.
    Incremental replanning should touch only the locally-affected nodes,
    unlike A* which has no memory and re-expands from scratch."""
    w = World(30, 30)
    d = DStarLite(w, (0, 0), (29, 29))
    assert d.get_path()[-1] == (29, 29)

    changed = [c for c in [(20, 5), (20, 6), (21, 5)] if w.add_obstacle(c)]
    d.reset_expansion_counter()
    d.notify_obstacles_changed(changed)
    incremental_expansions = d.node_expansions

    fresh_world = World(30, 30)
    for c in w.obstacles:
        fresh_world.add_obstacle(c)
    _, full_astar_expansions = astar_expansions(fresh_world, (0, 0), (29, 29))

    assert d.get_path()[-1] == (29, 29)
    assert all(c not in w.obstacles for c in d.get_path())
    assert incremental_expansions < full_astar_expansions
    print(f"incremental={incremental_expansions} full_astar={full_astar_expansions}")


def test_obstacle_spanning_the_entire_optimal_path_still_finds_detour():
    """Adversarial case: the obstacle blocks the whole previous route, so
    D* Lite must invalidate most of the chain it had cached. Correctness
    matters here even though the efficiency win from test above doesn't."""
    w = World(30, 30)
    d = DStarLite(w, (0, 0), (29, 29))
    assert d.get_path()[-1] == (29, 29)

    changed = [c for c in ((15, y) for y in range(25)) if w.add_obstacle(c)]
    d.notify_obstacles_changed(changed)
    path = d.get_path()
    assert path[-1] == (29, 29)
    assert all(c not in w.obstacles for c in path)


def test_no_path_when_fully_walled():
    w = World(5, 5)
    for y in range(5):
        w.add_obstacle((2, y))
    d = DStarLite(w, (0, 0), (4, 4))
    assert d.get_path() == []
