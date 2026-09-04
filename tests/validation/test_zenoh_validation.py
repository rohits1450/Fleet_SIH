"""Zenoh comms validation against Algovalidations/zenoh.png -- "the no
central server proof." Each robot below is a genuinely separate OS process
with its own zenoh session (see zenoh_worker.py); nothing here shares
Python memory across "robots" the way scenarios/fleet_sim.py's
InProcessBus does. This is real network pub/sub over loopback.

Kernel-level impairment (tc netem) needs root, which this sandbox doesn't
have non-interactively -- see zenoh_netem_manual.sh for the real version to
run yourself. Latency/loss tolerance here is validated with an
application-level fault-injection stand-in instead, which is clearly a
different (weaker) claim than kernel-level netem and is labeled as such.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import time

import pytest

from tests.validation.zenoh_worker import run_robot


def _spawn(agent_id, role, result_path, duration_s, position=(0.0, 0.0), **kwargs):
    p = multiprocessing.Process(
        target=run_robot, args=(agent_id, role, result_path, duration_s, position), kwargs=kwargs
    )
    p.start()
    return p


def _load(result_path):
    with open(result_path) as f:
        return json.load(f)


@pytest.fixture
def tmp_results(tmp_path):
    return lambda agent_id: str(tmp_path / f"{agent_id}.json")


# -- Scenario 1: Baseline mesh works ----------------------------------------
def test_baseline_mesh_all_robots_see_each_other_sub_100ms(tmp_results):
    n = 5
    ids = [f"r{i}" for i in range(n)]
    duration = 4.0
    procs = []
    for i, aid in enumerate(ids):
        role = "dispatcher" if i == 0 else "bidder"
        procs.append(_spawn(aid, role, tmp_results(aid), duration, position=(i * 5.0, 0.0)))
    for p in procs:
        p.join(timeout=duration + 15)
        assert p.exitcode == 0, f"robot process {p} did not exit cleanly (exitcode={p.exitcode})"

    results = {aid: _load(tmp_results(aid)) for aid in ids}
    max_latency = 0.0
    for aid, r in results.items():
        others = set(ids) - {aid}
        seen = set(r["poses_seen_from"].keys())
        assert others <= seen, f"{aid} never saw poses from {others - seen}"
        if r["pose_latencies_ms"]:
            max_latency = max(max_latency, max(r["pose_latencies_ms"]))

    print(f"[baseline mesh] {n} real zenoh sessions, all mutually visible, max_pose_latency={max_latency:.1f}ms")
    assert max_latency < 100.0, f"pose latency {max_latency:.1f}ms exceeds the 100ms baseline bound"


# -- Scenario 4 & 5: Partition demo + reconnection --------------------------
def test_partition_fleet_continues_and_reconnection_resyncs_without_duplication(tmp_results):
    duration = 9.0
    r0 = _spawn("r0", "dispatcher", tmp_results("r0"), duration, position=(0.0, 0.0),
                n_tasks=6, task_interval_s=0.6)
    r1 = _spawn("r1", "bidder", tmp_results("r1"), duration, position=(1.0, 0.0))
    r2 = _spawn("r2", "bidder", tmp_results("r2_before"), duration, position=(2.0, 0.0))
    r3 = _spawn("r3", "bidder", tmp_results("r3"), duration, position=(3.0, 0.0))

    # Simulate the Wi-Fi dead zone: r2's process (and its zenoh session with
    # it) is killed abruptly mid-run -- SIGTERM well before its natural
    # completion, so it never gets a chance to close cleanly or write a
    # result, the way a real dropped connection would look.
    time.sleep(3.5)
    r2.terminate()
    r2.join(timeout=5)
    assert not os.path.exists(tmp_results("r2_before")), (
        "r2 wrote a result before being killed -- didn't actually simulate an abrupt mid-run kill"
    )

    # Reconnection: r2 comes back online partway through the remaining
    # window, as a brand new process/session (never seen r0/r1/r3's state
    # from before it died).
    time.sleep(2.0)
    r2_again = _spawn("r2", "bidder", tmp_results("r2_after"), 3.0, position=(2.0, 0.0))

    for p in (r0, r1, r3, r2_again):
        p.join(timeout=duration + 15)
        assert p.exitcode == 0, f"process did not exit cleanly (exitcode={p.exitcode})"

    r0_result = _load(tmp_results("r0"))
    r1_result = _load(tmp_results("r1"))
    r3_result = _load(tmp_results("r3"))
    r2_after_result = _load(tmp_results("r2_after"))

    # The fleet must not have hung waiting on r2: r0's dispatch loop and
    # r1/r3's bidding must have kept going the whole time.
    assert r0_result["messages_sent"] > 0
    tasks_won_by = {}
    for r in (r0_result, r1_result, r3_result, r2_after_result):
        for tid, winner in r["winning_agents"].items():
            tasks_won_by.setdefault(tid, set()).add(winner)

    inconsistent = {tid: winners for tid, winners in tasks_won_by.items() if len(winners) > 1}
    print(f"[partition + reconnection] tasks seen: {len(tasks_won_by)}, "
          f"r1 bundle={r1_result['bundle']}, r3 bundle={r3_result['bundle']}, "
          f"r2(after reconnect) bundle={r2_after_result['bundle']}, "
          f"cross-agent inconsistent winners: {inconsistent}")
    assert not inconsistent, f"agents disagree on task winners after partition+reconnect: {inconsistent}"

    all_tasks = set(r0_result["known_task_ids"])
    assert all_tasks, "dispatcher never announced anything -- test setup invalid"
    won_tasks = {tid for tid, winners in tasks_won_by.items() if next(iter(winners)) is not None}
    assert len(won_tasks) >= len(all_tasks) - 1, (
        f"too many tasks left unclaimed after the fleet recovered: {all_tasks - won_tasks}"
    )


# -- Scenario 2: Latency injection (application-level stand-in for tc netem)
def test_latency_injection_still_converges_just_slower(tmp_results):
    """~100ms one-way delay injected at the publish call in each process.
    Not kernel netem (see module docstring) -- validates the coordination
    logic doesn't assume near-instant delivery, not the actual network."""
    duration = 6.0
    delay = 0.1
    ids = ["r0", "r1", "r2"]
    procs = [
        _spawn(aid, "dispatcher" if i == 0 else "bidder", tmp_results(aid), duration,
               position=(i * 5.0, 0.0), inject_delay_s=delay, n_tasks=3, task_interval_s=1.0)
        for i, aid in enumerate(ids)
    ]
    for p in procs:
        p.join(timeout=duration + 15)
        assert p.exitcode == 0, f"process crashed under injected latency (exitcode={p.exitcode})"

    results = {aid: _load(tmp_results(aid)) for aid in ids}
    for aid, r in results.items():
        others = set(ids) - {aid}
        seen = set(r["poses_seen_from"].keys())
        assert others <= seen, f"{aid} never converged under latency -- missing {others - seen}"
    winners = {tid: w for r in results.values() for tid, w in r["winning_agents"].items()}
    disagreements = {
        tid: {r["winning_agents"].get(tid) for r in results.values() if tid in r["winning_agents"]}
        for tid in winners
    }
    stale_state_bugs = {tid: ws for tid, ws in disagreements.items() if len(ws) > 1}
    print(f"[latency injection] {delay*1000:.0f}ms app-level delay, all agents still converged, "
          f"stale-state disagreements: {stale_state_bugs}")
    assert not stale_state_bugs, f"stale/inconsistent state under latency: {stale_state_bugs}"


# -- Scenario 3: Packet loss --------------------------------------------------
def test_packet_loss_cbba_still_reaches_consistent_state(tmp_results):
    """~15% of publishes dropped at the source (app-level stand-in for
    `tc netem loss`, see module docstring). CBBA's anti-entropy resync
    (core/allocation/cbba.py) is exactly what's meant to survive this."""
    duration = 7.0
    loss_rate = 0.15
    ids = ["r0", "r1", "r2", "r3"]
    procs = [
        _spawn(aid, "dispatcher" if i == 0 else "bidder", tmp_results(aid), duration,
               position=(i * 5.0, 0.0), inject_loss_rate=loss_rate, n_tasks=5, task_interval_s=0.6,
               anti_entropy_interval=10)
        for i, aid in enumerate(ids)
    ]
    for p in procs:
        p.join(timeout=duration + 15)
        assert p.exitcode == 0, f"process crashed under injected packet loss (exitcode={p.exitcode})"

    results = {aid: _load(tmp_results(aid)) for aid in ids}
    winners = {}
    for r in results.values():
        for tid, w in r["winning_agents"].items():
            winners.setdefault(tid, set()).add(w)
    double_booked = {tid: ws for tid, ws in winners.items() if len(ws) > 1}
    print(f"[packet loss] {loss_rate:.0%} drop rate, {len(winners)} tasks tracked, "
          f"double-booked: {double_booked}")
    assert not double_booked, f"tasks double-booked under packet loss: {double_booked}"
