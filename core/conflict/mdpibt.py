"""Priority-inheritance-style conflict resolution for continuously-moving
robots, using Karma for the pairwise yield decision (see karma.py).

This is a continuous-space simplification of MD-PIBT rather than its exact
grid-reservation algorithm: instead of reserving next-cells one discrete
step at a time, it looks at which robot pairs are close together and
closing distance (about to contest the same stretch of aisle) and decides,
every tick, which one yields (drops its preferred velocity to a crawl so
NH-ORCA lets the other one through cleanly instead of both slowing
symmetrically toward a stall). The yield relation is still exactly what
MD-PIBT's priority inheritance is *for*: a directed "waits for" edge
between agents. Chains of these edges are tracked as a dependency graph,
and a genuine cycle in it (A waits for B, B for C, C for A) is a real
deadlock -- broken here by forcing the lowest-karma agent in the cycle to
proceed, with the cycle's lifetime logged as the deadlock resolution time.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from core.conflict.karma import KarmaLedger

Vec2 = tuple[float, float]


@dataclass
class RobotView:
    agent_id: str
    position: Vec2
    velocity: Vec2


@dataclass
class ConflictResolver:
    karma: KarmaLedger
    conflict_radius: float = 1.4
    # Once two robots are already packed this close, arbitrate regardless of
    # whether either is still "closing" (dot(rel_pos, rel_vel) < 0). Without
    # this, a pileup that has already slowed everyone toward zero velocity
    # stops registering as a conflict at all -- relative velocity is ~0, not
    # negative -- so nobody ever gets picked to proceed, and the resolver
    # goes silent exactly when it's needed most.
    stuck_radius: float = 0.9
    # Once a pairwise conflict is decided, hold it for this many resolve()
    # calls before it's eligible to be re-decided. Without this, two
    # stationary robots re-run resolve_pair every single tick; since each
    # call pays 1 karma to the yielder and takes 1 from the winner, the
    # "who has lower cost" comparison flips on literally every call, so the
    # yield assignment alternates every tick and neither robot ever gets
    # priority for long enough to actually move -- a decision that looks
    # fine in any single-tick snapshot but is pure flicker over time.
    #
    # With equal deltas (the common case, since callers rarely pass custom
    # per-agent deltas) and payment=1 exactly offsetting tau=0.5,
    # resolve_pair provably flips back to a tie -- and then the same winner
    # again -- every time it's re-called, so any fixed hold just becomes a
    # slow, perfectly regular metronome instead of a fast flicker; it does
    # not, on its own, guarantee either robot clears the contested spot
    # before priority flips back. Swept empirically against the full fleet
    # sim (20 seeds x 150s, tracking worst-case continuous stall and
    # collisions): the relationship is NOT monotonic with hold length --
    # 1.5s and 2.5s were markedly worse than both 1.0s and 2.0s, i.e. this
    # is resonating with the sim's other periodic timers (stall/recovery,
    # congestion/detour), not a simple "longer is safer" tradeoff. 1.0s
    # (60 ticks) was the clear empirical best (worst stall 2.1s, zero
    # collisions across the swept seeds) and is what's set here; re-sweep
    # this value if STALL_THRESHOLD_S, RECOVERY_BURST_S, or
    # CONGESTION_TIMEOUT_S in scenarios/fleet_sim.py ever change.
    decision_hold_ticks: int = 60

    dependency_edges: list[tuple[str, str]] = field(default_factory=list)
    _deadlock_start_tick: dict[frozenset, int] = field(default_factory=dict)
    resolved_deadlock_durations: list[int] = field(default_factory=list)
    _pair_decisions: dict[frozenset, tuple[int, str, str]] = field(default_factory=dict)  # pair -> (tick, winner, yielder)

    def resolve(self, robots: dict[str, RobotView], tick: int) -> dict[str, bool]:
        """Returns {agent_id: should_yield} for this tick."""
        should_yield = {rid: False for rid in robots}
        self.dependency_edges = []
        seen_pairs: set[frozenset] = set()

        ids = list(robots.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                ra, rb = robots[a], robots[b]
                d = _dist(ra.position, rb.position)
                if d > self.conflict_radius:
                    continue
                if not (_closing(ra, rb) or d < self.stuck_radius):
                    continue

                pair_key = frozenset((a, b))
                seen_pairs.add(pair_key)
                cached = self._pair_decisions.get(pair_key)
                if cached is not None and tick - cached[0] < self.decision_hold_ticks:
                    _, winner, yielder = cached
                else:
                    winner, yielder = self.karma.resolve_pair(a, b)
                    self._pair_decisions[pair_key] = (tick, winner, yielder)

                should_yield[yielder] = True
                self.dependency_edges.append((yielder, winner))

        # A pair that's drifted out of conflict range has nothing left to
        # hold -- forget it, so a later re-approach starts fresh rather than
        # replaying a stale decision from an unrelated earlier encounter.
        for key in list(self._pair_decisions):
            if key not in seen_pairs and tick - self._pair_decisions[key][0] >= self.decision_hold_ticks:
                del self._pair_decisions[key]

        cycles = _find_cycles(self.dependency_edges)
        still_active = set()
        for cycle in cycles:
            key = frozenset(cycle)
            still_active.add(key)
            if key not in self._deadlock_start_tick:
                self._deadlock_start_tick[key] = tick
            # Break the deadlock: the agent that has yielded least so far
            # (lowest karma) is forced through regardless of the pairwise
            # calls above.
            proceed_agent = min(cycle, key=lambda rid: self.karma.balance(rid))
            should_yield[proceed_agent] = False

        for key in list(self._deadlock_start_tick):
            if key not in still_active:
                duration = tick - self._deadlock_start_tick.pop(key)
                self.resolved_deadlock_durations.append(duration)

        return should_yield


def _dist(a: Vec2, b: Vec2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _closing(ra: RobotView, rb: RobotView) -> bool:
    rel_pos = (rb.position[0] - ra.position[0], rb.position[1] - ra.position[1])
    rel_vel = (rb.velocity[0] - ra.velocity[0], rb.velocity[1] - ra.velocity[1])
    return (rel_pos[0] * rel_vel[0] + rel_pos[1] * rel_vel[1]) < 0


def _find_cycles(edges: list[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, list[str]] = defaultdict(list)
    for y, w in edges:
        graph[y].append(w)

    found: list[list[str]] = []
    seen_keys: set[frozenset] = set()

    def dfs(node: str, path: list[str], on_path: set[str]) -> None:
        for nxt in graph.get(node, []):
            if nxt in on_path:
                idx = path.index(nxt)
                cycle = path[idx:]
                key = frozenset(cycle)
                if key not in seen_keys:
                    seen_keys.add(key)
                    found.append(cycle)
            else:
                dfs(nxt, path + [nxt], on_path | {nxt})

    for start in list(graph.keys()):
        dfs(start, [start], {start})
    return found
