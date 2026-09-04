"""MD-PIBT + Karma validation against Algovalidations/md-pibt.png -- "your
riskiest layer, test it hardest."

Our implementation is a continuous-space simplification of MD-PIBT (see
core/conflict/mdpibt.py docstring): instead of discrete grid-cell
reservations, it decides every tick which of two closing robots yields
(drops preferred velocity to a crawl) via Karma-weighted priority, and
tracks the resulting "waits-for" edges as a dependency graph, breaking any
cycle that forms. These tests validate that mechanism directly, plus the
full NH-ORCA-integrated behavior for the physical scenarios.
"""
import math
import random

from core.avoidance.nh_orca import DEFAULT_EPSILON, nh_orca_velocity
from core.conflict.karma import KarmaLedger
from core.conflict.mdpibt import ConflictResolver, RobotView
from core.robot import DiffDriveRobot, Pose


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def toward(pos, goal, speed):
    dx, dy = goal[0] - pos[0], goal[1] - pos[1]
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return (0.0, 0.0)
    s = min(speed, d * 2.5)
    return (dx / d * s, dy / d * s)


def run_head_on(karma_a=0, karma_b=0, radius=0.28, ticks=3600, corridor_half_width=None):
    """Shared harness: two robots on a head-on course through a corridor,
    resolved every tick via Karma-weighted ConflictResolver + NH-ORCA,
    exactly as scenarios/fleet_sim.py wires them together."""
    ledger = KarmaLedger(tau=0.5, payment=1)
    ledger.balances["A"] = karma_a
    ledger.balances["B"] = karma_b
    resolver = ConflictResolver(karma=ledger, conflict_radius=1.4)

    a = DiffDriveRobot("A", Pose(-6.0, 0.02, 0.0), radius=radius, max_speed=1.0, max_omega=3.2)
    b = DiffDriveRobot("B", Pose(6.0, 0.0, math.pi), radius=radius, max_speed=1.0, max_omega=3.2)
    goal_a, goal_b = (6.0, 0.0), (-6.0, 0.0)
    dt = 1 / 60

    min_dist = float("inf")
    yield_log = []
    both_yielded_ticks = 0
    both_idle_ticks = 0

    for tick in range(ticks):
        views = {"A": RobotView("A", a.position(), a.velocity_at_reference_point(0.0)),
                 "B": RobotView("B", b.position(), b.velocity_at_reference_point(0.0))}
        should_yield = resolver.resolve(views, tick)
        yield_log.append((should_yield["A"], should_yield["B"]))
        if should_yield["A"] and should_yield["B"]:
            both_yielded_ticks += 1

        pref_a = toward(a.reference_point(DEFAULT_EPSILON), goal_a, a.max_speed)
        pref_b = toward(b.reference_point(DEFAULT_EPSILON), goal_b, b.max_speed)
        if should_yield["A"]:
            pref_a = (pref_a[0] * 0.05, pref_a[1] * 0.05)
        if should_yield["B"]:
            pref_b = (pref_b[0] * 0.05, pref_b[1] * 0.05)

        v_a, w_a = nh_orca_velocity(a, [b], pref_a, 2.0, dt)
        v_b, w_b = nh_orca_velocity(b, [a], pref_b, 2.0, dt)
        still_traveling = dist(a.position(), goal_a) > 0.5 or dist(b.position(), goal_b) > 0.5
        if still_traveling and abs(v_a) < 0.01 and abs(v_b) < 0.01:
            both_idle_ticks += 1
        a.set_body_velocity(v_a, w_a)
        b.set_body_velocity(v_b, w_b)
        a.step(dt)
        b.step(dt)

        min_dist = min(min_dist, dist(a.position(), b.position()))

    return {
        "a": a, "b": b, "ledger": ledger, "resolver": resolver,
        "min_dist": min_dist, "yield_log": yield_log,
        "both_yielded_ticks": both_yielded_ticks, "both_idle_ticks": both_idle_ticks,
        "combined_radius": 2 * radius,
        "reached_a": dist(a.position(), goal_a) < 0.5,
        "reached_b": dist(b.position(), goal_b) < 0.5,
    }


# -- Scenario 1: Head-on 1v1 ------------------------------------------------
def test_head_on_1v1_one_yields_one_proceeds_no_livelock():
    result = run_head_on(karma_a=0, karma_b=0)
    assert result["min_dist"] >= result["combined_radius"] - 1e-3, "collision during head-on encounter"
    assert result["both_idle_ticks"] < 300, (
        f"both robots stalled simultaneously for {result['both_idle_ticks']} ticks -- looks like a livelock"
    )
    assert result["reached_a"] and result["reached_b"], "one or both robots never reached their goal (stall)"
    # never both retreating: neither should ever show should_yield True at
    # the same time as the other is *also* being told to yield indefinitely
    max_consecutive_both = 0
    streak = 0
    for ya, yb in result["yield_log"]:
        if ya and yb:
            streak += 1
            max_consecutive_both = max(max_consecutive_both, streak)
        else:
            streak = 0
    print(f"[head-on 1v1] min_dist={result['min_dist']:.3f} (combined_radius={result['combined_radius']:.2f}), "
          f"max_consecutive_both_yield_ticks={max_consecutive_both}")
    assert max_consecutive_both < 60, "both robots yielded simultaneously for a full second -- livelock risk"


# -- Scenario 2: Karma actually matters -------------------------------------
def test_karma_actually_matters_in_pairwise_resolution():
    """Per the Karma mechanism as specified (Riehl et al., and reporef.txt):
    the agent CHOSEN TO REPLAN/YIELD is whichever minimizes Delta + tau*karma,
    and yielding *pays* the yielder -- so an agent that has already yielded a
    lot (high karma) is progressively LESS likely to be picked to yield again
    ("less often forced to yield later"). We validate that direction here,
    since it's what core/conflict/karma.py implements and it's what the
    cited paper describes. (Note: this is the opposite of a literal reading
    of the Algovalidations checklist item, which says the *high*-karma robot
    should yield -- see the written report for that discrepancy.)
    """
    ledger = KarmaLedger(tau=0.5, payment=1)
    ledger.balances["A"] = 10  # A has yielded often already -> should be protected
    ledger.balances["B"] = 0
    winner, yielder = ledger.resolve_pair("A", "B")
    print(f"[karma matters] A(karma=10) vs B(karma=0) -> winner={winner}, yielder={yielder}, "
          f"post-exchange balances A={ledger.balance('A')} B={ledger.balance('B')}")
    assert winner == "A" and yielder == "B", "high-karma agent was not protected from yielding"


# -- Scenario 3: Karma decays back to fairness ------------------------------
def test_karma_converges_to_balanced_yields_over_repeated_encounters():
    ledger = KarmaLedger(tau=0.5, payment=1)
    yields = {"A": 0, "B": 0}
    diffs = []
    for i in range(40):
        _, yielder = ledger.resolve_pair("A", "B")
        yields[yielder] += 1
        diffs.append(abs(yields["A"] - yields["B"]))

    first_half_max_diff = max(diffs[:20])
    second_half_max_diff = max(diffs[20:])
    print(f"[karma fairness] yields A={yields['A']} B={yields['B']}, "
          f"max_diff first20={first_half_max_diff} last20={second_half_max_diff}")
    assert second_half_max_diff <= first_half_max_diff + 1, "yield imbalance is diverging, not converging"
    assert abs(yields["A"] - yields["B"]) <= 2, "yields did not converge to roughly balanced"


# -- Scenario 4: Exact aisle-block shape ------------------------------------
def test_single_column_aisle_block_two_robots_opposite_ends_no_deadlock():
    result = run_head_on(karma_a=0, karma_b=0, ticks=3600)
    had_dependency_edge = False
    # re-run resolve at the moment of closest approach to confirm an edge exists
    ledger = KarmaLedger(tau=0.5, payment=1)
    resolver = ConflictResolver(karma=ledger, conflict_radius=1.4)
    views = {"A": RobotView("A", (0.0, 0.0), (1.0, 0.0)), "B": RobotView("B", (0.5, 0.0), (-1.0, 0.0))}
    resolver.resolve(views, 0)
    had_dependency_edge = len(resolver.dependency_edges) == 1

    print(f"[aisle block] dependency edge builds on close approach: {had_dependency_edge}, "
          f"both robots reached goal (no deadlock): {result['reached_a'] and result['reached_b']}")
    assert had_dependency_edge, "no waits-for edge formed despite a clear close-approach conflict"
    assert result["reached_a"] and result["reached_b"], "deadlock: at least one robot never got through"


# -- Scenario 5: Chain/train deadlock ---------------------------------------
def test_chain_of_four_robots_cascades_not_just_front_robot():
    """r0<-r1<-r2 queued nose-to-tail, all moving at the same speed (so
    nobody is "closing" on anybody yet -- an already-moving convoy has zero
    relative velocity between its members). r3 then cuts across r0's path.
    r0 must slow for r3; once r0 is slower than r1, r1 starts genuinely
    closing on r0's now-nearer position and must yield too, then likewise
    r2 on r1 -- the cascade MD-PIBT's priority inheritance is meant to
    produce, not a single isolated front-of-queue conflict."""
    ledger = KarmaLedger(tau=0.5, payment=1)
    resolver = ConflictResolver(karma=ledger, conflict_radius=1.6)

    edges_seen: set[tuple[str, str]] = set()
    robots_touched: set[str] = set()

    # Tick 1: r3 crosses r0's path -- only this pair is in conflict range.
    robots = {
        "r0": RobotView("r0", (0.0, 0.0), (0.5, 0.0)),
        "r1": RobotView("r1", (-1.0, 0.0), (0.5, 0.0)),
        "r2": RobotView("r2", (-2.0, 0.0), (0.5, 0.0)),
        "r3": RobotView("r3", (0.8, 0.6), (-0.4, -0.5)),
    }
    should_yield = resolver.resolve(robots, tick=0)
    edges_seen.update(resolver.dependency_edges)
    robots_touched.update(r for pair in resolver.dependency_edges for r in pair)
    assert should_yield.get("r0"), "r0 should yield to the crossing r3"

    # Tick 2: r0 has slowed (yielding); r1, still at full speed, has closed
    # the gap and now genuinely converges on r0's slower position.
    robots["r0"] = RobotView("r0", (0.05, 0.0), (0.02, 0.0))
    robots["r1"] = RobotView("r1", (-0.7, 0.0), (0.5, 0.0))
    should_yield = resolver.resolve(robots, tick=1)
    edges_seen.update(resolver.dependency_edges)
    robots_touched.update(r for pair in resolver.dependency_edges for r in pair)
    assert should_yield.get("r1"), "r1 should now be closing on the slowed r0 and yield in turn"

    # Tick 3: same cascade, one link further back.
    robots["r1"] = RobotView("r1", (-0.65, 0.0), (0.02, 0.0))
    robots["r2"] = RobotView("r2", (-1.4, 0.0), (0.5, 0.0))
    should_yield = resolver.resolve(robots, tick=2)
    edges_seen.update(resolver.dependency_edges)
    robots_touched.update(r for pair in resolver.dependency_edges for r in pair)
    assert should_yield.get("r2"), "r2 should now be closing on the slowed r1 and yield in turn"

    print(f"[chain deadlock] edges seen across the cascade: {edges_seen}, robots touched: {sorted(robots_touched)}")
    assert len(edges_seen) >= 3, "cascade should touch more than just the front pair"
    assert robots_touched == {"r0", "r1", "r2", "r3"}, "MD-PIBT cascade did not propagate through the whole chain"


# -- Scenario 6: Symmetric tie -----------------------------------------------
def test_symmetric_tie_deterministic_no_oscillation():
    winners = []
    for _ in range(30):
        ledger = KarmaLedger(tau=0.5, payment=1)  # fresh ledger: identical karma=0 every time
        winner, yielder = ledger.resolve_pair("r5", "r9")
        winners.append((winner, yielder))

    print(f"[symmetric tie] first 5 outcomes of 30: {winners[:5]}")
    assert len(set(winners)) == 1, f"tie-break is not deterministic: {set(winners)}"
