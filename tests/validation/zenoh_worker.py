"""One robot process for the real-Zenoh validation harness (Algovalidations/
zenoh.png). Each invocation opens its own zenoh session -- a genuine
separate peer, not a thread sharing state -- publishes timestamped pose
samples and participates in CBBA bidding over the network, then writes a
JSON summary of what it observed to `result_path` so the test process (which
can't reach into another OS process's memory) can check the outcome.

Runnable directly too: `python -m tests.validation.zenoh_worker r0 dispatcher /tmp/out.json 5`
"""
from __future__ import annotations

import json
import sys
import time

from core.allocation.cbba import CBBAAgent, Task
from core.comms.zenoh_bus import ZenohBus

POSE_TOPIC_TMPL = "fleet/{agent}/pose"
POSE_SUBSCRIBE_PATTERN = "fleet/*/pose"


def run_robot(
    agent_id: str,
    role: str,  # "dispatcher" announces tasks; anything else just bids
    result_path: str,
    duration_s: float,
    position: tuple[float, float] = (0.0, 0.0),
    n_tasks: int = 5,
    task_interval_s: float = 0.3,
    inject_delay_s: float = 0.0,
    inject_loss_rate: float = 0.0,
    anti_entropy_interval: int = 50,
) -> None:
    bus = ZenohBus(inject_delay_s=inject_delay_s, inject_loss_rate=inject_loss_rate)
    pose_latencies_ms: list[float] = []
    poses_seen: dict[str, int] = {}

    def on_pose(topic: str, payload) -> None:
        sender, sent_ts = payload
        if sender == agent_id:
            return
        pose_latencies_ms.append((time.time() - sent_ts) * 1000.0)
        poses_seen[sender] = poses_seen.get(sender, 0) + 1

    bus.subscribe(POSE_SUBSCRIBE_PATTERN, on_pose)
    cbba = CBBAAgent(agent_id=agent_id, bus=bus, get_position=lambda: position, max_bundle=3,
                      anti_entropy_interval=anti_entropy_interval)

    time.sleep(0.4)  # let zenoh discovery settle before relying on delivery

    start = time.time()
    next_task_at = start + task_interval_s
    tasks_announced = 0
    while time.time() - start < duration_s:
        bus.publish(POSE_TOPIC_TMPL.format(agent=agent_id), (agent_id, time.time()))
        cbba.step_tick()  # drives CBBA's anti-entropy resync -- without this, a bid
        # dropped under packet loss has no retry and agents can disagree forever.
        if role == "dispatcher" and tasks_announced < n_tasks and time.time() >= next_task_at:
            task = Task(
                task_id=f"{agent_id}-task{tasks_announced}",
                pickup=(tasks_announced, 0), dropoff=(tasks_announced, 1), reward=100.0,
            )
            cbba.announce_task(task)
            tasks_announced += 1
            next_task_at = time.time() + task_interval_s
        time.sleep(0.05)

    # Drain window: keep ticking (not just sleeping) so any bid still
    # unresolved from earlier loss gets several more independent
    # anti-entropy attempts before we report final state.
    drain_until = time.time() + 1.5
    while time.time() < drain_until:
        cbba.step_tick()
        time.sleep(0.05)

    summary = {
        "agent_id": agent_id,
        "zenoh_id": bus.zenoh_id(),
        "pose_latencies_ms": pose_latencies_ms,
        "poses_seen_from": poses_seen,
        "bundle": cbba.bundle,
        "winning_agents": cbba.winning_agents,
        "winning_bids": cbba.winning_bids,
        "timestamps": cbba.timestamps,
        "known_task_ids": sorted(cbba.tasks.keys()),
        "messages_sent": bus.message_count(),
    }
    with open(result_path, "w") as f:
        json.dump(summary, f)
    bus.close()


if __name__ == "__main__":
    _, agent_id, role, result_path, duration_s = sys.argv[:5]
    run_robot(agent_id, role, result_path, float(duration_s))
