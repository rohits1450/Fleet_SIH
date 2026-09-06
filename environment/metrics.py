"""Fleet-wide metrics: task completion times (makespan), collision events,
and simple accumulators the renderer reads each frame. CBBA message counts,
D* Lite node-expansion counts, and Karma balances/variance live on their
own objects (CBBAAgent, DStarLite, KarmaLedger) and are read directly --
this class only tracks things nothing else naturally owns.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FleetMetrics:
    task_durations: list[float] = field(default_factory=list)
    tasks_completed: int = 0
    collision_count: int = 0
    start_tick: int = 0

    def task_completed(self, duration_s: float) -> None:
        self.tasks_completed += 1
        self.task_durations.append(duration_s)

    def register_collision(self) -> None:
        self.collision_count += 1

    def mean_task_duration(self) -> float:
        if not self.task_durations:
            return 0.0
        return sum(self.task_durations) / len(self.task_durations)

    def makespan(self, now_s: float) -> float:
        return now_s - self.start_tick
