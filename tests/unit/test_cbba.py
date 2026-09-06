from algorithms.task_allocation.cbba import CBBAAgent, Task
from communication.inprocess import InProcessBus


def make_fleet(n_agents, positions, max_bundle=2):
    bus = InProcessBus()
    agents = [
        CBBAAgent(agent_id=f"r{i}", bus=bus, get_position=(lambda p=positions[i]: p), max_bundle=max_bundle)
        for i in range(n_agents)
    ]
    return bus, agents


def test_two_agents_split_two_far_apart_tasks():
    bus, agents = make_fleet(2, [(0, 0), (20, 20)])
    t1 = Task("t1", pickup=(1, 1), dropoff=(2, 2), reward=100.0)
    t2 = Task("t2", pickup=(19, 19), dropoff=(18, 18), reward=100.0)

    agents[0].announce_task(t1)
    agents[0].announce_task(t2)
    for a in agents:
        a.try_build_bundle()

    # winning_agents must agree across the fleet (converged consensus)
    for tid in ("t1", "t2"):
        winners = {a.winning_agents.get(tid) for a in agents}
        assert len(winners) == 1, f"agents disagree on winner of {tid}: {winners}"

    # the near agent should win the near task
    assert agents[0].winning_agents["t1"] == "r0"
    assert agents[1].winning_agents["t2"] == "r1"

    # no task double-claimed in any agent's own bundle
    all_bundled = [tid for a in agents for tid in a.bundle]
    assert len(all_bundled) == len(set(all_bundled))


def test_outbid_releases_downstream_bundle_entries():
    bus, agents = make_fleet(2, [(0, 0), (0.1, 0.1)], max_bundle=2)
    # nearly co-located agents bidding on the same two tasks: whichever
    # bids second on a task already in the other's bundle must yield, and
    # cascade-release anything added after it.
    t1 = Task("t1", pickup=(1, 0), dropoff=(1, 1), reward=50.0)
    t2 = Task("t2", pickup=(2, 0), dropoff=(2, 1), reward=50.0)
    agents[0].announce_task(t1)
    agents[0].announce_task(t2)

    agents[0].try_build_bundle()
    agents[1].try_build_bundle()

    for tid in ("t1", "t2"):
        assert agents[0].winning_agents[tid] == agents[1].winning_agents[tid]

    combined_bundle = agents[0].bundle + agents[1].bundle
    assert sorted(combined_bundle) == sorted(set(combined_bundle))
    assert set(combined_bundle) == {"t1", "t2"}


def test_ed_cbba_publishes_far_fewer_messages_than_periodic():
    bus, agents = make_fleet(3, [(0, 0), (10, 0), (20, 0)], max_bundle=1)
    for i in range(6):
        agents[0].announce_task(Task(f"t{i}", pickup=(i, 0), dropoff=(i, 1), reward=100.0))
    for a in agents:
        a.try_build_bundle()

    ticks = 50
    for a in agents:
        for _ in range(ticks):
            a.step_tick()

    total_ed_messages = sum(a.messages_sent for a in agents)
    total_periodic_messages = sum(a.periodic_message_cost for a in agents)
    assert total_ed_messages < total_periodic_messages
    reduction = 1 - total_ed_messages / total_periodic_messages
    assert reduction > 0.5, f"expected >50% reduction, got {reduction:.1%}"
