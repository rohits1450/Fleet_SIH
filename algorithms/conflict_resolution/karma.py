"""Karma: a fairness mechanism for pairwise conflict resolution (Riehl et
al., arXiv:2604.07970). Each agent holds a non-tradeable integer balance.
On a conflict, the agent chosen to yield/replan is the one that minimizes
the composite cost Delta + tau*karma (its own replanning cost plus its
current karma weighted by tau, ~0.5 balances fairness against efficiency).
The yielder is then paid: karma flows from the agent who got priority to
the agent who yielded. Over repeated conflicts this means an agent that
has yielded often accumulates karma, making it progressively *more*
expensive (and so less likely) to be picked to yield again -- the
replanning burden equalizes across the fleet instead of always falling on
whichever agent happens to be in the way.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class KarmaLedger:
    tau: float = 0.5
    payment: int = 1
    balances: dict[str, int] = field(default_factory=dict)

    def balance(self, agent_id: str) -> int:
        return self.balances.get(agent_id, 0)

    def composite_cost(self, agent_id: str, delta: float) -> float:
        return delta + self.tau * self.balance(agent_id)

    def resolve_pair(self, agent_a: str, agent_b: str, delta_a: float = 1.0, delta_b: float = 1.0) -> tuple[str, str]:
        """Returns (winner, yielder). The yielder is paid `payment` karma
        by the winner."""
        cost_a = self.composite_cost(agent_a, delta_a)
        cost_b = self.composite_cost(agent_b, delta_b)
        # Lower composite cost is "chosen to replan" (yields); ties broken
        # toward agent_a for determinism.
        yielder, winner = (agent_a, agent_b) if cost_a <= cost_b else (agent_b, agent_a)
        self.balances[yielder] = self.balance(yielder) + self.payment
        self.balances[winner] = self.balance(winner) - self.payment
        return winner, yielder

    def variance(self) -> float:
        if len(self.balances) < 2:
            return 0.0
        return statistics.pvariance(self.balances.values())
