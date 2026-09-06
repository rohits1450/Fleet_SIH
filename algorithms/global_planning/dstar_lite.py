"""D* Lite (Koenig & Likhachev, 2002) on an 8-connected grid.

Maintains g(s) (cost-so-far-to-goal estimate) and rhs(s) (one-step
lookahead value) per node. On a local map change, only nodes that become
locally inconsistent (g != rhs) are re-expanded, instead of recomputing the
whole search like A* would. `node_expansions` counts pops off the priority
queue so callers can instrument and compare replanning cost against A*.

The search runs backward from the goal so that when the robot moves one
step, only the heuristic (relative to the new start) shifts, and a single
`km` offset keeps stale keys in the queue comparable to fresh ones.
"""
from __future__ import annotations

import heapq
import itertools
import math
from typing import Optional

from environment.grid_world import Cell, World

INF = math.inf
_KEY_EPS = 1e-7


def octile_heuristic(a: Cell, b: Cell) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


class DStarLite:
    def __init__(self, world: World, start: Cell, goal: Cell):
        self.world = world
        self.start = start
        self.goal = goal
        self._last_start = start
        self.km = 0.0

        self.g: dict[Cell, float] = {}
        self.rhs: dict[Cell, float] = {}
        self._entry_finder: dict[Cell, tuple] = {}
        self._counter = itertools.count()
        self._queue: list[tuple[tuple[float, float], int, Cell]] = []

        self.node_expansions = 0  # cumulative, reset via reset_expansion_counter()
        self.last_expanded: list[Cell] = []  # cells popped during the most recent compute_shortest_path()

        self.rhs[self.goal] = 0.0
        self._push(self.goal)
        self.compute_shortest_path()

    # -- priority queue helpers -------------------------------------------------
    def _g(self, s: Cell) -> float:
        return self.g.get(s, INF)

    def _rhs(self, s: Cell) -> float:
        return self.rhs.get(s, INF)

    def _calculate_key(self, s: Cell) -> tuple[float, float]:
        m = min(self._g(s), self._rhs(s))
        return (m + octile_heuristic(self.start, s) + self.km, m)

    def _push(self, s: Cell) -> None:
        self._remove(s)
        count = next(self._counter)
        entry = (self._calculate_key(s), count, s)
        self._entry_finder[s] = entry
        heapq.heappush(self._queue, entry)

    def _remove(self, s: Cell) -> None:
        # Lazy deletion: drop the lookup entry only. Stale heap entries are
        # skipped in _top_key/_pop by checking against _entry_finder.
        self._entry_finder.pop(s, None)

    def _top_key(self) -> tuple[float, float]:
        while self._queue:
            key, count, s = self._queue[0]
            if self._entry_finder.get(s) != (key, count, s):
                heapq.heappop(self._queue)
                continue
            return key
        return (INF, INF)

    def _pop(self) -> Optional[Cell]:
        while self._queue:
            key, count, s = heapq.heappop(self._queue)
            if self._entry_finder.get(s) != (key, count, s):
                continue  # stale entry, superseded by a later _push
            del self._entry_finder[s]
            return s
        return None

    # -- core algorithm -----------------------------------------------------
    def _cost(self, a: Cell, b: Cell) -> float:
        if not self.world.is_free(a) or not self.world.is_free(b):
            return INF
        if b not in set(self.world.neighbors(a)):
            return INF
        return self.world.edge_cost(a, b)

    def _preds(self, s: Cell):
        # symmetric 8-connected grid: predecessors == neighbors of a free cell
        yield from self.world.neighbors(s)

    def _update_vertex(self, u: Cell) -> None:
        if u != self.goal:
            best = INF
            for s in self._preds(u):
                cost = self._cost(u, s)
                if cost >= INF:
                    continue
                val = cost + self._g(s)
                if val < best:
                    best = val
            self.rhs[u] = best
        self._remove(u)
        if self._g(u) != self._rhs(u):
            self._push(u)

    def compute_shortest_path(self) -> None:
        self.last_expanded = []
        while self._queue:
            top_key = self._top_key()
            start_key = self._calculate_key(self.start)
            # Tolerant, not strict: a change landing exactly on the previously
            # optimal path telescopes to the same key at every node along it
            # (old cost-to-goal + heuristic to start == start's own key), so
            # float rounding alone can push top_key a hair past start_key.
            # Treating that as "greater" stops the cascade before it reaches
            # start, leaving stale g-values. Compare with an epsilon so a true
            # tie (or noise around one) still gets popped and propagated.
            if top_key[0] > start_key[0] + _KEY_EPS and self._rhs(self.start) == self._g(self.start):
                break
            u = self._pop()
            if u is None:
                break
            self.node_expansions += 1
            self.last_expanded.append(u)
            k_old = top_key
            k_new = self._calculate_key(u)
            if k_old < k_new:
                self._push(u)
            elif self._g(u) > self._rhs(u):
                self.g[u] = self._rhs(u)
                for s in self._preds(u):
                    self._update_vertex(s)
            else:
                self.g[u] = INF
                for s in itertools.chain(self._preds(u), (u,)):
                    self._update_vertex(s)

    def reset_expansion_counter(self) -> None:
        self.node_expansions = 0

    # -- public API for the sim loop ----------------------------------------
    def update_start(self, new_start: Cell) -> None:
        self.km += octile_heuristic(self._last_start, new_start)
        self._last_start = new_start
        self.start = new_start

    def notify_obstacles_changed(self, changed_cells: list[Cell]) -> None:
        """Call after mutating world.obstacles for the given cells."""
        for cell in changed_cells:
            self._update_vertex(cell)
            for n in self.world.neighbors(cell):
                self._update_vertex(n)
            # also re-check neighbors that may now be blocked from cell (cell itself
            # may be an obstacle so world.neighbors(cell) yields nothing; scan its
            # 8-neighborhood directly regardless of free/blocked state)
            x, y = cell
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                n = (x + dx, y + dy)
                if self.world.in_bounds(n):
                    self._update_vertex(n)
        self.compute_shortest_path()

    def get_path(self) -> list[Cell]:
        if self._rhs(self.start) >= INF:
            return []
        path = [self.start]
        current = self.start
        seen = {current}
        while current != self.goal:
            best_cell = None
            best_val = INF
            for s in self.world.neighbors(current):
                val = self.world.edge_cost(current, s) + self._g(s)
                if val < best_val:
                    best_val = val
                    best_cell = s
            if best_cell is None or best_cell in seen or best_val >= INF:
                return path  # no path / local cycle guard
            path.append(best_cell)
            seen.add(best_cell)
            current = best_cell
            if len(path) > self.world.width * self.world.height:
                break
        return path


def astar_expansions(world: World, start: Cell, goal: Cell) -> tuple[list[Cell], int]:
    """Full from-scratch A* search, used only as a baseline to compare
    node-expansion counts against D* Lite's incremental replanning."""
    open_set: list[tuple[float, int, Cell]] = []
    counter = itertools.count()
    g = {start: 0.0}
    came_from: dict[Cell, Cell] = {}
    heapq.heappush(open_set, (octile_heuristic(start, goal), next(counter), start))
    closed: set[Cell] = set()
    expansions = 0

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)
        expansions += 1
        if current == goal:
            break
        for n in world.neighbors(current):
            tentative = g[current] + world.edge_cost(current, n)
            if tentative < g.get(n, INF):
                g[n] = tentative
                came_from[n] = current
                heapq.heappush(open_set, (tentative + octile_heuristic(n, goal), next(counter), n))

    if goal not in g:
        return [], expansions
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path, expansions
