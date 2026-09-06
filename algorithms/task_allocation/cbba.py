"""CBBA (Consensus-Based Bundle Algorithm, Choi/Brunet/How 2009) and its
event-driven variant, ED-CBBA (Sao et al., arXiv:2509.06481).

Each robot runs one `CBBAAgent`. Two phases:
  1. Bundle construction: greedily add the task with the highest diminishing
     marginal gain (reward minus the extra route length it adds at its best
     insertion point), as long as that gain beats the best bid anyone else
     has published for it.
  2. Consensus: on receiving a neighbor's (bid, winning-agent, timestamp)
     table, adopt whichever claim is more recent/higher-bid per task; if
     that outbids a task in our own bundle, release it and everything added
     to the bundle after it (their marginal costs assumed the released task
     was still there).

ED-CBBA's contribution is entirely about *when* to publish: standard CBBA
broadcasts every round regardless of change; here `build_bundle()` only
publishes when the bundle actually grew, and consensus only re-triggers
construction when a release actually freed capacity -- publishing is a
side effect of a real state change, not a clock tick. `periodic_message_cost`
lets the caller tally what a fixed-period broadcaster would have sent over
the same run, for the message-count comparison the paper reports (up to
52% fewer messages, no loss of allocation quality).
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Optional

from communication.bus import MessageBus
from environment.grid_world import Cell

TASKS_TOPIC = "fleet/tasks/announce"
BIDS_TOPIC_TMPL = "fleet/{agent}/bids"
BIDS_SUBSCRIBE_PATTERN = "fleet/*/bids"


@dataclass
class Task:
    task_id: str
    pickup: Cell
    dropoff: Cell
    reward: float = 100.0
    created_tick: int = 0


def _dist(a: Cell, b: Cell) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _route_positions(start: Cell, path: list[str], tasks: dict[str, Task]) -> list[Cell]:
    positions = [start]
    for tid in path:
        t = tasks[tid]
        positions.append(t.pickup)
        positions.append(t.dropoff)
    return positions


def _route_cost(positions: list[Cell]) -> float:
    return sum(_dist(a, b) for a, b in zip(positions, positions[1:]))


@dataclass
class CBBAAgent:
    agent_id: str
    bus: MessageBus
    get_position: "callable[[], Cell]"
    max_bundle: int = 2
    # Optional: bidding assumes an agent that wins a task can go execute it.
    # If that's not currently true (e.g. physically boxed in by obstacles),
    # wire this to say so -- otherwise a stuck agent can win a task purely on
    # distance score, immediately have to reject it, and re-win the very
    # retry it just rejected, forever.
    can_participate: "callable[[], bool] | None" = None

    tasks: dict[str, Task] = field(default_factory=dict)
    bundle: list[str] = field(default_factory=list)     # insertion order
    path: list[str] = field(default_factory=list)       # visiting order
    winning_bids: dict[str, float] = field(default_factory=dict)
    winning_agents: dict[str, Optional[str]] = field(default_factory=dict)
    timestamps: dict[str, int] = field(default_factory=dict)

    tick: int = 0
    messages_sent: int = 0
    periodic_message_cost: int = 0  # what a fixed-period broadcaster would have sent, for comparison
    anti_entropy_interval: int = 50

    # Zenoh (unlike InProcessBus) delivers subscriber callbacks on its own
    # background thread, concurrently with whatever the owning process's
    # main loop is doing. Every method that reads-then-mutates bundle/path/
    # winning_bids etc. is a critical section under real Zenoh; reentrant
    # because announce_task/release_task/_on_bids all call try_build_bundle.
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.bus.subscribe(TASKS_TOPIC, self._on_task_announced)
        self.bus.subscribe(BIDS_SUBSCRIBE_PATTERN, self._on_bids)

    # -- task discovery -------------------------------------------------
    def announce_task(self, task: Task) -> None:
        with self._lock:
            self.tasks[task.task_id] = task
            self.bus.publish(TASKS_TOPIC, task)
            # publish() only fires *other* agents' subscriptions usefully
            # here -- our own registry already has the task, so our
            # _on_task_announced no-ops. Without this, whichever agent
            # announces never bids on its own announcements.
            self.try_build_bundle()

    def _on_task_announced(self, _topic: str, task: Task) -> None:
        with self._lock:
            if task.task_id not in self.tasks:
                self.tasks[task.task_id] = task
                self.try_build_bundle()

    # -- bundle construction ----------------------------------------------
    def try_build_bundle(self) -> bool:
        """Greedily grow the bundle. Returns True (and publishes) iff it
        actually grew -- the event-driven trigger."""
        with self._lock:
            if self.can_participate is not None and not self.can_participate():
                return False
            grew = False
            while len(self.bundle) < self.max_bundle:
                best_task, best_pos, best_score = None, None, float("-inf")
                start = self.get_position()
                base_positions = _route_positions(start, self.path, self.tasks)
                base_cost = _route_cost(base_positions)

                for tid, task in self.tasks.items():
                    if tid in self.bundle:
                        continue
                    current_best_bid = self.winning_bids.get(tid, 0.0)
                    for pos in range(len(self.path) + 1):
                        trial_path = self.path[:pos] + [tid] + self.path[pos:]
                        trial_cost = _route_cost(_route_positions(start, trial_path, self.tasks))
                        score = task.reward - (trial_cost - base_cost)
                        if score > best_score and score > current_best_bid:
                            best_score, best_task, best_pos = score, tid, pos

                if best_task is None:
                    break

                self.bundle.append(best_task)
                self.path.insert(best_pos, best_task)
                self.winning_bids[best_task] = best_score
                self.winning_agents[best_task] = self.agent_id
                self.timestamps[best_task] = self.tick
                grew = True

            if grew:
                self._publish_bids()
            return grew

    def release_task(self, task_id: str) -> None:
        """Called when a task completes (pickup+dropoff done): clears it out
        and immediately tries to fill the freed slot from whatever's already
        waiting, rather than sitting idle until the next unrelated
        announcement happens to trigger a rescan."""
        with self._lock:
            if task_id in self.bundle:
                self.bundle.remove(task_id)
            if task_id in self.path:
                self.path.remove(task_id)
            self.try_build_bundle()

    def next_task(self) -> Optional[Task]:
        with self._lock:
            return self.tasks[self.path[0]] if self.path else None

    def _publish_bids(self) -> None:
        self.messages_sent += 1
        table = {
            tid: (self.winning_bids[tid], self.winning_agents[tid], self.timestamps[tid])
            for tid in self.winning_bids
        }
        self.bus.publish(BIDS_TOPIC_TMPL.format(agent=self.agent_id), (self.agent_id, table))

    # -- consensus ----------------------------------------------------------
    def _on_bids(self, _topic: str, msg: tuple[str, dict[str, tuple[float, Optional[str], int]]]) -> None:
        sender_id, table = msg
        if sender_id == self.agent_id:
            return
        with self._lock:
            released_from_bundle = False

            for tid, (inc_bid, inc_agent, inc_ts) in table.items():
                cur_bid = self.winning_bids.get(tid, float("-inf"))
                cur_agent = self.winning_agents.get(tid)
                cur_ts = self.timestamps.get(tid, -1)

                # Bid value is the primary ordering, not recency: this makes
                # the per-task record a max-register (like a CRDT), so it
                # converges to the same value everywhere regardless of what
                # order lossy, out-of-order delivery happens to bring
                # messages in. Using "newer timestamp wins" as anything
                # other than a last-resort tie-break lets a later, WORSE bid
                # stomp an earlier, better one for whichever peers happen to
                # hear it first -- a genuine split-brain under packet loss,
                # not just slower convergence.
                #
                # An exact bid tie (routinely real -- e.g. evenly-spaced
                # tasks give two agents identical marginal insertion cost)
                # can also tie on timestamp, since both bids may have been
                # computed on the same tick. Without a THIRD, order-
                # independent tie-break, each observer just keeps whichever
                # claim it happened to hear first forever: a stable split
                # where every peer is locally self-consistent but the fleet
                # as a whole never agrees. Breaking remaining ties by agent
                # id gives every observer the same answer regardless of
                # arrival order.
                if cur_agent is None:
                    trust_incoming = True
                elif inc_bid > cur_bid + 1e-9:
                    trust_incoming = True
                elif abs(inc_bid - cur_bid) <= 1e-9:
                    if inc_ts > cur_ts:
                        trust_incoming = True
                    elif inc_ts == cur_ts:
                        trust_incoming = (inc_agent or "") < (cur_agent or "")
                    else:
                        trust_incoming = False
                else:
                    trust_incoming = False

                if not trust_incoming:
                    continue

                was_mine = cur_agent == self.agent_id
                self.winning_bids[tid] = inc_bid
                self.winning_agents[tid] = inc_agent
                self.timestamps[tid] = inc_ts

                if was_mine and inc_agent != self.agent_id and tid in self.bundle:
                    idx = self.bundle.index(tid)
                    released = self.bundle[idx:]
                    self.bundle = self.bundle[:idx]
                    self.path = [t for t in self.path if t not in released]
                    released_from_bundle = True

            if released_from_bundle:
                self.try_build_bundle()

    def step_tick(self) -> None:
        with self._lock:
            self.tick += 1
            # What a fixed-period broadcaster would have cost, purely for
            # the ED-CBBA vs periodic-CBBA message-count comparison shown in
            # the UI.
            self.periodic_message_cost += 1
            # Anti-entropy: a single dropped bid message under a lossy link
            # has no retry, so two agents can permanently disagree about who
            # won a task (each blind to the other's conflicting claim). A
            # full-table resync this rarely doesn't undermine ED-CBBA's
            # message savings -- it's a background heartbeat for
            # robustness, not the event-driven bidding path -- but
            # guarantees eventual convergence despite loss.
            if self.winning_bids and self.tick % self.anti_entropy_interval == 0:
                self._publish_bids()
