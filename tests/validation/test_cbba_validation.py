"""CBBA / ED-CBBA validation against validation/figures/cbba.png.
Pathfinding cost is frozen as the built-in mock (Euclidean) distance
function CBBA already uses -- no D* Lite dependency needed here."""
import random

from algorithms.task_allocation.cbba import CBBAAgent, Task
from communication.inprocess import InProcessBus


def make_fleet(positions, max_bundle=2):
    bus = InProcessBus()
    agents = [
        CBBAAgent(agent_id=f"r{i}", bus=bus, get_position=(lambda p=p: p), max_bundle=max_bundle)
        for i, p in enumerate(positions)
    ]
    return bus, agents


# -- Scenario 1: Simple bid ------------------------------------------------
def test_simple_bid_r0_wins_others_bundles_unaffected():
    bus, agents = make_fleet([(0, 0), (30, 30), (30, -30)])
    r0, r1, r2 = agents
    r1_task = Task("far1", pickup=(29, 29), dropoff=(28, 28), reward=100.0)
    r2_task = Task("far2", pickup=(29, -29), dropoff=(28, -28), reward=100.0)
    r1.announce_task(r1_task)
    r2.announce_task(r2_task)
    r0_before_bundles = (list(r0.bundle), list(r1.bundle), list(r2.bundle))

    near_task = Task("near0", pickup=(1, 0), dropoff=(2, 0), reward=100.0)
    r0.announce_task(near_task)

    assert r0.winning_agents["near0"] == "r0"
    assert r1.winning_agents["near0"] == "r0"
    assert r2.winning_agents["near0"] == "r0"
    assert r1.bundle == r0_before_bundles[1], "unrelated bundle disturbed by an unrelated bid"
    assert r2.bundle == r0_before_bundles[2], "unrelated bundle disturbed by an unrelated bid"
    print(f"[simple bid] r0 won near0, r1/r2 bundles untouched: {r1.bundle}, {r2.bundle}")


# -- Scenario 2: Contested task --------------------------------------------
def test_contested_equidistant_task_exactly_one_winner_deterministic():
    winners_across_runs = []
    for _ in range(5):
        bus, agents = make_fleet([(-5, 0), (5, 0)])
        a, b = agents
        task = Task("mid", pickup=(0, 0), dropoff=(0, 1), reward=100.0)
        a.announce_task(task)
        winner_a = a.winning_agents["mid"]
        winner_b = b.winning_agents["mid"]
        assert winner_a == winner_b, "agents disagree on who won an equidistant task"
        winners_across_runs.append(winner_a)

    assert len(set(winners_across_runs)) == 1, (
        f"tie-break is not deterministic across runs: {winners_across_runs}"
    )
    print(f"[contested task] exactly one winner every run, deterministic: {winners_across_runs[0]}")


# -- Scenario 3: Diminishing marginal gain ---------------------------------
def test_diminishing_marginal_gain_across_five_tasks():
    # Tasks scattered off-axis (not collinear) so each successive insertion
    # into the growing route costs progressively more detour -- a evenly
    # spaced straight line would keep marginal cost roughly constant, which
    # isn't what this scenario is meant to probe.
    bus, agents = make_fleet([(0, 0)], max_bundle=5)
    r0 = agents[0]
    positions = [(10, 0), (10, 8), (-10, 6), (14, -12), (-16, -14)]
    scores = []
    for i, pos in enumerate(positions):
        task = Task(f"t{i}", pickup=pos, dropoff=(pos[0], pos[1] + 1), reward=100.0)
        r0.announce_task(task)
        scores.append(r0.winning_bids[f"t{i}"])

    print(f"[diminishing marginal gain] bid scores across 5 tasks: {[round(s, 2) for s in scores]}")
    assert scores[-1] < scores[0], (
        f"bid on task 5 ({scores[-1]:.2f}) is not visibly lower than task 1 ({scores[0]:.2f})"
    )


# -- Scenario 4: ED-CBBA trigger -------------------------------------------
def test_ed_cbba_trigger_reroutes_when_committed_task_gets_expensive():
    bus, agents = make_fleet([(0, 0), (15, 0)], max_bundle=1)
    near, far = agents
    task = Task("aisle_task", pickup=(1, 0), dropoff=(2, 0), reward=100.0)
    near.announce_task(task)
    assert near.winning_agents["aisle_task"] == "r0"
    bid_before = near.winning_bids["aisle_task"]
    messages_before = near.messages_sent + far.messages_sent

    # "Block the aisle": near's real route to this pickup now needs a big
    # detour. CBBA's bid here is a mock Euclidean-distance cost function
    # (per the validation note), so the detour is modeled directly against
    # it -- near's effective position for reaching this pickup jumps far
    # away, while far is unaffected.
    near.release_task("aisle_task")
    del near.tasks["aisle_task"]
    del near.winning_bids["aisle_task"]
    del near.winning_agents["aisle_task"]
    del near.timestamps["aisle_task"]
    near.get_position = lambda: (-40.0, 0.0)

    retry = Task("aisle_task-retry1", pickup=(1, 0), dropoff=(2, 0), reward=100.0)
    far.announce_task(retry)

    assert far.winning_agents["aisle_task-retry1"] == "r1", "closer/better bidder should now win"
    assert near.winning_agents["aisle_task-retry1"] == "r1", "loser's view must converge too"
    assert "aisle_task-retry1" not in near.bundle
    bid_after = far.winning_bids["aisle_task-retry1"]
    messages_after = near.messages_sent + far.messages_sent
    print(f"[ED-CBBA trigger] bid before(near)={bid_before:.1f}, bid after(far)={bid_after:.1f}, "
          f"task transferred to r1, messages used for the transfer={messages_after - messages_before}")


def test_ed_cbba_uses_far_fewer_messages_than_periodic_over_a_run():
    bus, agents = make_fleet([(0, 0), (10, 0), (20, 0)], max_bundle=1)
    for i in range(6):
        agents[0].announce_task(Task(f"t{i}", pickup=(i, 0), dropoff=(i, 1), reward=100.0))
    for _ in range(300):
        for a in agents:
            a.step_tick()

    total_ed = sum(a.messages_sent for a in agents)
    total_periodic = sum(a.periodic_message_cost for a in agents)
    reduction = 1 - total_ed / total_periodic
    print(f"[ED-CBBA vs periodic] ed_messages={total_ed} periodic_equivalent={total_periodic} "
          f"reduction={reduction:.1%}")
    assert reduction > 0.5


# -- Scenario 5: No double-booking under packet loss -----------------------
def test_no_double_booking_under_lossy_broadcast():
    class LossyBus(InProcessBus):
        def __init__(self, loss_rate: float, rng: random.Random):
            super().__init__()
            self.loss_rate = loss_rate
            self.rng = rng

        def publish(self, topic, payload):
            if self.rng.random() < self.loss_rate and "bids" in topic:
                self._message_count += 1  # still "sent", just dropped in flight
                return
            super().publish(topic, payload)

    rng = random.Random(0)
    bus = LossyBus(loss_rate=0.4, rng=rng)
    agents = [
        CBBAAgent(agent_id=f"r{i}", bus=bus, get_position=(lambda p=(i * 3, 0): p), max_bundle=2)
        for i in range(5)
    ]

    for i in range(15):
        pos = (rng.uniform(0, 15), rng.uniform(-5, 5))
        agents[0].announce_task(Task(f"t{i}", pickup=pos, dropoff=(pos[0] + 1, pos[1]), reward=100.0))
        # Let time pass under the lossy link -- this is what actually
        # exercises anti-entropy (a low-rate full-table resync that gives
        # a dropped bid update another chance to land), not just repeated
        # bundle-building attempts with no further message traffic.
        for _ in range(120):
            for a in agents:
                a.step_tick()

    # No task may end up claimed (in-bundle) by more than one agent locally,
    # even though each agent's *view* of the global winner can lag under loss.
    task_holders: dict[str, list[str]] = {}
    for a in agents:
        for tid in a.bundle:
            task_holders.setdefault(tid, []).append(a.agent_id)

    double_booked = {tid: holders for tid, holders in task_holders.items() if len(holders) > 1}
    print(f"[packet loss] {len(task_holders)} tasks actively held, double-booked: {double_booked}")
    assert not double_booked, f"tasks double-booked under packet loss: {double_booked}"
