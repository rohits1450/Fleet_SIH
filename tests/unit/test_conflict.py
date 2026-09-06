import statistics

from algorithms.conflict_resolution.karma import KarmaLedger
from algorithms.conflict_resolution.mdpibt import ConflictResolver, RobotView


def test_karma_yield_counts_equalize_over_repeated_conflicts():
    ledger = KarmaLedger(tau=0.5, payment=1)
    yields = {"a": 0, "b": 0}
    for _ in range(40):
        winner, yielder = ledger.resolve_pair("a", "b")
        yields[yielder] += 1

    # Without karma, the same (lower-id / arbitrary) agent would yield every
    # time. With karma feeding back into the cost, yields should roughly
    # balance rather than concentrate on one agent.
    assert abs(yields["a"] - yields["b"]) <= 2
    assert ledger.balance("a") + ledger.balance("b") == 0


def test_karma_prefers_yielding_the_lower_balance_agent():
    ledger = KarmaLedger(tau=0.5, payment=1)
    ledger.balances["a"] = 5  # a has already yielded a lot -> protected
    ledger.balances["b"] = 0
    winner, yielder = ledger.resolve_pair("a", "b")
    assert yielder == "b"
    assert winner == "a"


def test_three_way_cycle_is_detected_and_broken():
    karma = KarmaLedger(tau=0.5, payment=1)
    resolver = ConflictResolver(karma=karma, conflict_radius=5.0)

    # Three robots at the corners of a triangle, each heading toward the
    # next one's position -> all pairwise-closing -> should form a 3-cycle.
    robots = {
        "a": RobotView("a", position=(0.0, 0.0), velocity=(1.0, 0.0)),
        "b": RobotView("b", position=(1.0, 0.0), velocity=(-0.5, 0.87)),
        "c": RobotView("c", position=(0.5, 0.87), velocity=(-0.5, -0.87)),
    }

    should_yield = resolver.resolve(robots, tick=1)

    # A genuine deadlock must not freeze everyone: at least one robot has to
    # be allowed through.
    assert not all(should_yield.values()), "cycle-breaking failed to release any agent"
    assert any(should_yield.values()), "expected at least one yield edge to form the cycle"


def test_deadlock_duration_is_recorded_once_resolved():
    karma = KarmaLedger(tau=0.5, payment=1)
    resolver = ConflictResolver(karma=karma, conflict_radius=5.0)
    cycle_robots = {
        "a": RobotView("a", position=(0.0, 0.0), velocity=(1.0, 0.0)),
        "b": RobotView("b", position=(1.0, 0.0), velocity=(-0.5, 0.87)),
        "c": RobotView("c", position=(0.5, 0.87), velocity=(-0.5, -0.87)),
    }
    for tick in range(5):
        resolver.resolve(cycle_robots, tick=tick)

    # Robots disperse: no longer closing on each other -> cycle should clear.
    dispersed = {
        "a": RobotView("a", position=(0.0, 0.0), velocity=(0.0, 0.0)),
        "b": RobotView("b", position=(10.0, 0.0), velocity=(0.0, 0.0)),
        "c": RobotView("c", position=(0.0, 10.0), velocity=(0.0, 0.0)),
    }
    resolver.resolve(dispersed, tick=5)
    assert len(resolver.resolved_deadlock_durations) == 1
    assert resolver.resolved_deadlock_durations[0] >= 0


def test_stationary_pileup_still_gets_arbitrated_not_silently_ignored():
    """Once a pileup has already slowed everyone to near-zero velocity,
    nobody is "closing" (dot(rel_pos, rel_vel) < 0) any more -- without a
    stuck-radius fallback, the resolver would go silent and leave every
    robot with should_yield=False forever (each thinks it has the right of
    way, nobody actually has priority, matching the "pushing each other but
    nobody moves" bug)."""
    karma = KarmaLedger(tau=0.5, payment=1)
    resolver = ConflictResolver(karma=karma, conflict_radius=1.4, stuck_radius=0.9)

    packed = {
        "a": RobotView("a", position=(0.0, 0.0), velocity=(0.001, 0.0)),
        "b": RobotView("b", position=(0.3, 0.0), velocity=(-0.001, 0.0)),
        "c": RobotView("c", position=(0.0, 0.3), velocity=(0.0, -0.001)),
    }
    should_yield = resolver.resolve(packed, tick=0)

    assert resolver.dependency_edges, "a stationary pileup produced no arbitration at all"
    assert not all(should_yield.values()), "everyone yielding simultaneously is the exact stuck state"
    assert any(not v for v in should_yield.values()), "someone must be granted priority to proceed"
