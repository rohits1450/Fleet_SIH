"""Grid world model shared by every layer.

Holds the static occupancy grid (obstacles, shelves, aisles) plus named
pickup/dropoff points. Continuous robot poses live in the same coordinate
frame as the grid (cell size = 1.0 world unit) so planners, avoidance, and
the renderer all agree on geometry without conversion layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


Cell = tuple[int, int]


@dataclass
class World:
    width: int
    height: int
    obstacles: set[Cell] = field(default_factory=set)
    pickup_points: dict[str, Cell] = field(default_factory=dict)
    dropoff_points: dict[str, Cell] = field(default_factory=dict)
    # A diagonal move's straight-line path passes right by the shared corner
    # of its two "side" cells. Lenient (default) only forbids it when BOTH
    # sides are blocked -- fine for a point robot. A robot with real radius
    # can still clip that corner when just ONE side is blocked, so a strict
    # world requires BOTH sides free before allowing the diagonal at all.
    strict_diagonal_corners: bool = False

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and cell not in self.obstacles

    def add_obstacle(self, cell: Cell) -> bool:
        """Returns True if this actually changed the map."""
        if self.in_bounds(cell) and cell not in self.obstacles:
            self.obstacles.add(cell)
            return True
        return False

    def remove_obstacle(self, cell: Cell) -> bool:
        if cell in self.obstacles:
            self.obstacles.discard(cell)
            return True
        return False

    def toggle_obstacle(self, cell: Cell) -> bool:
        if cell in self.obstacles:
            self.obstacles.discard(cell)
        else:
            if not self.in_bounds(cell):
                return False
            self.obstacles.add(cell)
        return True

    def neighbors(self, cell: Cell) -> Iterable[Cell]:
        x, y = cell
        # 8-connected grid; diagonal moves cost more (see edge_cost).
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            n = (x + dx, y + dy)
            if self.is_free(n):
                if dx != 0 and dy != 0:
                    side_a, side_b = self.is_free((x + dx, y)), self.is_free((x, y + dy))
                    if self.strict_diagonal_corners:
                        if not (side_a and side_b):
                            continue
                    elif not side_a and not side_b:
                        continue
                yield n

    def edge_cost(self, a: Cell, b: Cell) -> float:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return 1.4142135623730951 if dx and dy else 1.0
