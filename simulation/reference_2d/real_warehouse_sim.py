"""Integrated fleet simulator running on a real, ROS map_server-style
warehouse map (YAML + image) instead of the procedural pod grid in
fleet_sim.py.

Same four layers as fleet_sim.py -- D* Lite + NH-ORCA + CBBA/ED-CBBA +
Karma-weighted MD-PIBT conflict resolution -- wired up identically. What's
different is entirely in how the warehouse itself is built: the World comes
from environment/mapio.py's map loader, pickup points are auto-placed at aisle
cells next to detected shelving, dropoff points at detected wall gaps
(loading docks), and every real-world (meters, m/s) constant is converted
into this map's grid-cell units via `m()`, since the source map's
resolution (downsampled to CELL_SIZE_M meters/cell) generally isn't 1:1
with the "1 cell = 1 world unit" convention environment/grid_world.py assumes.

Run: python -m simulation.reference_2d.real_warehouse_sim
"""
from __future__ import annotations

import heapq
import math
import random
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import pygame

from algorithms.task_allocation.cbba import CBBAAgent, Task
from algorithms.local_planning.nh_orca import nh_orca_velocity
from communication.inprocess import InProcessBus
from algorithms.conflict_resolution.karma import KarmaLedger
from algorithms.conflict_resolution.mdpibt import ConflictResolver, RobotView
from environment.mapio import (
    boundary_gaps,
    classify_wall_and_shelf_components,
    find_clear_cells,
    inward_direction,
    line_cell,
    load_world_from_map,
    shelf_adjacent_cells,
)
from environment.metrics import FleetMetrics
from algorithms.global_planning.dstar_lite import DStarLite
from models.robot import DiffDriveRobot, Pose
from environment.grid_world import Cell, World

MAP_YAML = Path(__file__).resolve().parent.parent.parent / "maps" / "custom_warehouse_pedestrians_solid_shelves.yaml"
# Downsample from the map's native 0.05m/px resolution to this coarser
# planning-cell size -- native resolution would give D* Lite a ~206k-cell
# grid, far more than a per-tick replan needs.
CELL_SIZE_M = 0.2

_WORLD, _MAP_META = load_world_from_map(MAP_YAML, cell_size=CELL_SIZE_M)
_WALL_COMPONENTS, _SHELF_COMPONENTS = classify_wall_and_shelf_components(_WORLD)
_WALL_CELLS: set[Cell] = set().union(*_WALL_COMPONENTS) if _WALL_COMPONENTS else set()
_SHELF_CELLS: set[Cell] = set().union(*_SHELF_COMPONENTS) if _SHELF_COMPONENTS else set()

if _WALL_CELLS:
    _WALL_BBOX = (
        min(c[0] for c in _WALL_CELLS), min(c[1] for c in _WALL_CELLS),
        max(c[0] for c in _WALL_CELLS), max(c[1] for c in _WALL_CELLS),
    )
else:
    _WALL_BBOX = (0, 0, _WORLD.width - 1, _WORLD.height - 1)

GRID_W, GRID_H = _WORLD.width, _WORLD.height
TOP_BAR = 110
# This map's grid (97x134) is much bigger than the pod-grid demo's (30x18),
# so a fixed pixels-per-cell that fits the whole thing in one window makes
# individual cells tiny. Instead of a fixed CELL_PX, the window is
# resizable and the view supports pan (right-drag or arrow keys/WASD) and
# zoom (mouse wheel) -- see the Camera-related attributes on
# RealWarehouseFleetSim (self.zoom, self.cam_x, self.cam_y) and
# world_to_px/px_to_world/cell_rect below.
INITIAL_WINDOW_W, INITIAL_WINDOW_H = 1100, 800
DEFAULT_ZOOM = 18.0   # pixels per grid cell at startup
MIN_ZOOM, MAX_ZOOM = 3.0, 60.0
ZOOM_STEP = 1.15
PAN_SPEED_CELLS_PER_S = 20.0

DT = 1.0 / 60.0
TIME_HORIZON = 2.0
PICK_DROP_SECONDS = 0.6
TASK_SPAWN_INTERVAL_S = 3.0
MAX_PENDING_TASKS = 4
# CBBA's bid score is reward minus the route length a task adds (both in
# grid cells); fleet_sim.py's reward=100.0 was sized for its ~30x18 pod
# grid (diagonal ~35 cells). This map's grid is 97x134 (diagonal ~166
# cells) -- at reward=100, most tasks would cost more than they're worth,
# so no agent would ever bid on them. Scale up proportionally to this
# grid's diagonal, keeping the same reward-to-max-distance safety margin.
TASK_REWARD = 100.0 * math.hypot(GRID_W, GRID_H) / math.hypot(30, 18)
STALL_THRESHOLD_S = 1.0
RECOVERY_BURST_S = 0.4
RECOVERY_GAIN = 4.0  # control gain (rad/s per rad of heading error) -- dimensionless, no meters conversion
UNREACHABLE_REASSIGN_S = 3.0
CONGESTION_TIMEOUT_S = 1.2
DETOUR_COMMIT_S = 3.0
DETOUR_MAX_COST_RATIO = 1.6
NH_ORCA_STATIC_TIME_HORIZON = 0.4


def m(meters: float) -> float:
    """Convert a real-world meters quantity into this map's grid-cell units
    (environment/grid_world.py treats 1 cell = 1 world unit, but this map's cells are
    CELL_SIZE_M meters wide, not 1m)."""
    return meters / CELL_SIZE_M


# fleet_sim.py's pod-grid demo tuned every size constant below (robot radii,
# NH-ORCA epsilon, arrive/jam/conflict radii, ...) around a ~0.28-0.42m
# robot radius. This map's shelf aisles measure only 0.6-0.8m wide (see
# environment/mapio.py's gap detection), which those robots plus NH-ORCA's epsilon
# margin can't fit through at all -- not a tuning problem, a hard geometric
# one. ROBOT_SCALE shrinks the whole fleet (and every other size constant
# that was tuned relative to it) down to a footprint that clears these
# aisles with real margin, keeping every constant's original *ratio* to
# robot size intact rather than re-tuning each one from scratch.
ROBOT_SCALE = 0.3
# Robot *speed* doesn't need to shrink by the same factor as robot *size* --
# a smaller robot isn't inherently slower, and this map's real dimensions
# (~19x27m) make full-ROBOT_SCALE speeds (~0.35 m/s) painfully slow to
# watch: a single pickup-or-dropoff leg alone can take a minute or more,
# which reads as "frozen" even when it's just en route. SPEED_SCALE is
# tuned independently, faster than ROBOT_SCALE but still well under the
# pod-grid demo's original full speed (NH-ORCA's TIME_HORIZON is fixed in
# seconds, so a faster robot needs more physical distance to react in these
# still-narrow aisles -- validated empirically, see tests/validation).
SPEED_SCALE = 0.7


def sm(meters: float) -> float:
    """Convert one of fleet_sim.py's original tuned-for-bigger-robots meters
    value into this scenario's grid-cell units, applying ROBOT_SCALE."""
    return m(meters * ROBOT_SCALE)


def sm_speed(meters_per_s: float) -> float:
    """Like sm(), but for a speed (m/s) -- scaled by SPEED_SCALE instead of
    ROBOT_SCALE since a robot's speed isn't tied to its footprint."""
    return m(meters_per_s * SPEED_SCALE)


ARRIVE_THRESH = sm(0.22)
ADVANCE_THRESH = sm(0.45)
STALL_SPEED_THRESH = sm_speed(0.05)
JAM_SPEED_THRESH = sm_speed(0.05)
JAM_RADIUS = sm(1.2)
EPSILON = sm(0.18)          # NH-ORCA reference-point offset, see algorithms/local_planning/nh_orca.py
WHEEL_BASE = sm(0.4)
CONFLICT_RADIUS = sm(1.4)
STUCK_RADIUS = sm(0.9)

# One heavy-lift AMR (much larger, slower, harder to turn) alongside three
# standard drive units, same fleet composition as fleet_sim.py's demo --
# radii scaled by ROBOT_SCALE (aisle clearance), speeds by SPEED_SCALE
# (responsiveness).
ROBOT_SPECS = [
    {"radius": sm(0.42), "max_speed": sm_speed(0.85), "max_omega": 2.0, "label": "HEAVY"},
    {"radius": sm(0.28), "max_speed": sm_speed(1.15), "max_omega": 3.2, "label": "AMR"},
    {"radius": sm(0.28), "max_speed": sm_speed(1.15), "max_omega": 3.2, "label": "AMR"},
    {"radius": sm(0.28), "max_speed": sm_speed(1.15), "max_omega": 3.2, "label": "AMR"},
]
N_ROBOTS = len(ROBOT_SPECS)
ROBOT_COLORS = [
    (240, 200, 90), (90, 170, 240), (240, 130, 130), (140, 220, 120),
]

COLOR_BG = (18, 20, 24)
COLOR_FREE = (40, 44, 52)
COLOR_GRID_LINE = (28, 31, 37)
COLOR_POD = (196, 148, 66)
COLOR_POD_EDGE = (150, 108, 40)
COLOR_OBSTACLE = (200, 70, 70)
COLOR_TEXT = (225, 225, 230)
COLOR_DIM = (150, 155, 165)
COLOR_PICKUP = (90, 200, 120)
COLOR_DROPOFF = (220, 140, 90)
COLOR_YIELD_EDGE = (240, 210, 90)
COLOR_WARN = (240, 90, 90)
COLOR_CARGO = (180, 140, 90)
COLOR_CARGO_EDGE = (110, 82, 48)


def cell_center(c: Cell) -> tuple[float, float]:
    return (c[0] + 0.5, c[1] + 0.5)


def pos_to_cell(pos: tuple[float, float]) -> Cell:
    return (int(math.floor(pos[0])), int(math.floor(pos[1])))


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _path_cost(path: list[Cell]) -> float:
    return sum(dist(cell_center(a), cell_center(b)) for a, b in zip(path, path[1:]))


def toward(pos, goal, speed) -> tuple[float, float]:
    dx, dy = goal[0] - pos[0], goal[1] - pos[1]
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return (0.0, 0.0)
    s = min(speed, d * 2.5)
    return (dx / d * s, dy / d * s)


def angle_diff(target: float, current: float) -> float:
    return (target - current + math.pi) % (2 * math.pi) - math.pi


def _draw_dashed_line(screen, p1, p2, color, dash_len=6) -> None:
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1e-6:
        return
    steps = max(1, int(length // dash_len))
    for i in range(0, steps, 2):
        t0, t1 = i / steps, min(1.0, (i + 1) / steps)
        a = (x1 + (x2 - x1) * t0, y1 + (y2 - y1) * t0)
        b = (x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1)
        pygame.draw.line(screen, color, a, b, 1)


def _active_target(agent: "RobotAgent") -> Cell | None:
    """The cell this agent is currently working toward -- its task's
    pickup while inbound to or docked at pickup, its dropoff once carrying.
    None when idle. Used to draw an at-a-glance line + ring from robot to
    goal, distinct from the fine-grained D* Lite path outline."""
    if agent.task is None:
        return None
    if agent.state in (TaskState.TO_PICKUP, TaskState.PICKING):
        return agent.task.pickup
    if agent.state in (TaskState.TO_DROPOFF, TaskState.DROPPING):
        return agent.task.dropoff
    return None


# -- deriving task/spawn points from the map's geometry ----------------

# Direction a candidate pickup cell should be nudged to get *away* from the
# shelf it's adjacent to (deeper into the aisle) -- the opposite sense from
# mapio.inward_direction, which points from a wall into the building rather
# than from a shelf face into open floor.
_SHELF_OUTWARD: dict[str, Cell] = {"N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0)}


def _find_clear_approach(world: World, start: Cell, direction: Cell, max_steps: int = 4) -> Cell | None:
    """Nudge a shelf-adjacent cell outward, looking for one with real
    docking clearance -- not just "technically free", and not just "clear
    enough for a small AMR" either. A cell immediately touching a shelf is
    only ~0.1m from its surface at this map's resolution, tighter than any
    robot with real radius can actually stop at, and this fleet has more
    than one radius: a spot with just enough room for an AMR can still be
    too tight for the larger HEAVY unit, which would then stall on its
    final approach even though D* Lite's route (planned in ITS OWN
    per-radius inflated world) says the cell is a valid goal. Checking
    against _FLEET_CLEARANCE_WORLD -- inflated for the single LARGEST
    robot in ROBOT_SPECS -- means a point offered here is one every robot
    in the fleet can actually dock at, not just the smallest.

    Returns None (rather than the nearest free-but-tight cell) when no step
    within range clears that bar: a marginal fallback here isn't just a
    slightly worse docking spot, it's a cell whose entire neighborhood can
    end up obstacle in _inflate_world's per-robot planning copy (see
    RealWarehouseFleetSim._planning_world_for) -- D* Lite would then see a
    goal with literally no viable approach edge and fail every attempt
    forever, regardless of which robot gets the retry. Better to not offer
    this shelf side as a pickup point at all than promise an unreachable one."""
    dx, dy = direction
    for step in range(max_steps + 1):
        c = (start[0] + dx * step, start[1] + dy * step)
        if not world.is_free(c):
            return None
        if _FLEET_CLEARANCE_WORLD.is_free(c):
            return c
    return None


def _shelf_pickup_points(world: World, shelves: list[set[Cell]]) -> list[Cell]:
    """One pickup point per accessible side of each shelf component -- the
    aisle cell nearest that side's midpoint (mirroring how a robot would
    approach a warehouse shelf from whichever aisle reaches it), nudged out
    for standoff clearance. Shelf sides with no cell offering real
    clearance within range are skipped (see _find_clear_approach) rather
    than offered as a pickup point that would later prove unreachable."""
    points: list[Cell] = []
    for shelf in shelves:
        xs = [c[0] for c in shelf]
        ys = [c[1] for c in shelf]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        for side, cells in shelf_adjacent_cells(world, shelf).items():
            nearest = min(cells, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
            approach = _find_clear_approach(world, nearest, _SHELF_OUTWARD[side])
            if approach is not None:
                points.append(approach)
    return points


def _push_inward(world: World, side: str, line_coord: int, perp_coord: int, max_steps: int = 8) -> Cell | None:
    """Walk from a wall-line cell inward into the building, looking for a
    cell with real docking clearance for the largest robot in the fleet
    (see _find_clear_approach for why) -- so a dropoff point doesn't sit in
    a doorway pinch point that _inflate_world's per-robot planning copy
    could later isolate entirely. Returns None if nothing in range qualifies."""
    dx, dy = inward_direction(side)
    start = line_cell(side, line_coord, perp_coord)
    for step in range(max_steps):
        c = (start[0] + dx * step, start[1] + dy * step)
        if not world.in_bounds(c) or not world.is_free(c):
            continue
        if _FLEET_CLEARANCE_WORLD.is_free(c):
            return c
    return None


def _entrance_dropoff_points(
    world: World, wall_cells: set[Cell], points_per_wide_gap: int = 3, min_span_for_multi: int = 6
) -> list[Cell]:
    """Dropoff (loading-dock) points just inside each detected wall gap. A
    wide gap (a garage-door-style dock apron, not just a single door) gets
    several spread-out points instead of one, so traffic isn't forced
    through one exact cell."""
    points: list[Cell] = []
    if not wall_cells:
        return points
    for side, line_coord, c0, c1 in boundary_gaps(world, wall_cells):
        span = c1 - c0
        n = points_per_wide_gap if span >= min_span_for_multi else 1
        for i in range(n):
            # Spread across most of the gap's width (not just its middle
            # third) so points land several cells apart -- packed close
            # together, multiple robots parked at neighboring dropoff
            # points sit well within each other's NH-ORCA avoidance radius
            # and can stall each other out at the dock.
            frac = 0.5 if n == 1 else 0.1 + i * 0.8 / (n - 1)
            perp = round(c0 + frac * span)
            cell = _push_inward(world, side, line_coord, perp)
            if cell is not None:
                points.append(cell)
    return points


def _inflate_world(base: World, clearance: float, restore_connectivity: bool = True) -> World:
    """A copy of `base` with every free cell whose center is closer than
    `clearance` to any obstacle's surface also marked obstacle.

    algorithms/global_planning/dstar_lite.py has no notion of robot radius at all (see
    GAZEBO_INTEGRATION.md) -- it happily proposes the geometrically-shortest
    route through a single-cell-wide passage even when that's narrower than
    the robot planning it. NH-ORCA then can't actually realize that route
    (it correctly refuses to let the robot's real footprint through), and
    the two layers disagree forever: the planner keeps re-offering the same
    too-tight path, avoidance keeps stalling on it. Planning against a
    clearance-inflated copy of the map instead (one per distinct robot
    radius) makes D* Lite steer around passages this robot can't fit
    through in the first place, the standard "configuration space" fix.
    NH-ORCA/rendering/collision checks still use the real, uninflated
    world -- only route *planning* sees the inflated one.

    `restore_connectivity=False` skips _restore_connectivity below -- used
    by _FLEET_CLEARANCE_WORLD, where the point is deciding whether a cell
    is a *comfortable place to stop*, not whether a route may pass through
    it. Restoration correctly re-opens a tight cell that's the only real
    bridge between two regions (a robot must be allowed to transit it), but
    that says nothing about whether it's a good place to dock -- reusing
    the restored world here let pickup points land in exactly such
    bridges, still only ~0.1m from an obstacle on each side."""
    inflated = World(base.width, base.height, strict_diagonal_corners=base.strict_diagonal_corners)
    inflated.obstacles = set(base.obstacles)
    margin = int(math.ceil(clearance)) + 1
    to_add = []
    for y in range(base.height):
        for x in range(base.width):
            if (x, y) in base.obstacles:
                continue
            cx, cy = x + 0.5, y + 0.5
            for ox in range(x - margin, x + margin + 1):
                for oy in range(y - margin, y + margin + 1):
                    if (ox, oy) not in base.obstacles:
                        continue
                    nx = max(ox, min(cx, ox + 1.0))
                    ny = max(oy, min(cy, oy + 1.0))
                    if dist((cx, cy), (nx, ny)) < clearance:
                        to_add.append((x, y))
                        break
                else:
                    continue
                break
    for c in to_add:
        inflated.add_obstacle(c)
    if restore_connectivity:
        _restore_connectivity(inflated, base)
    return inflated


def _restore_connectivity(world: World, base: World) -> None:
    """Undo just enough inflation to guarantee every cell free in `base`
    can still reach every other one in `world`.

    Inflation is a per-cell local clearance check with no idea it can
    sever the map into disconnected pieces -- and it does: a robot resting
    at a real, valid position (after a pickup/dropoff pause, mid-detour,
    anywhere) can find itself on a cell whose entire neighborhood reads
    "too tight for me" even though nothing changed about whether it's
    actually standing there. D* Lite then can't plan any route AT ALL from
    that start, regardless of the goal -- confirmed by comparing a live
    stuck robot's planner against a from-scratch replan on the exact same
    world/start/goal, which failed identically. A planner that can't route
    from where the robot already, physically is isn't being safe, it's
    broken. This finds the cheapest bridge of inflated (not really
    obstacle) cells back to the main region for every isolated pocket and
    un-inflates exactly those, so connectivity always matches the real map's."""
    free = [(x, y) for x in range(world.width) for y in range(world.height) if world.is_free((x, y))]
    if not free:
        return
    visited: set[Cell] = set()
    components: list[list[Cell]] = []
    for start in free:
        if start in visited:
            continue
        stack, comp = [start], [start]
        visited.add(start)
        while stack:
            c = stack.pop()
            for n in world.neighbors(c):
                if n not in visited:
                    visited.add(n)
                    comp.append(n)
                    stack.append(n)
        components.append(comp)
    if len(components) <= 1:
        return

    components.sort(key=len, reverse=True)
    main = set(components[0])
    for island in components[1:]:
        # Dijkstra out from the island: stepping onto a cell that's already
        # free costs 0, one that's obstacle-in-`world`-but-free-in-`base`
        # (revertible) costs 1, and a genuine `base` obstacle is impassable
        # -- cheapest path to `main` is the minimal bridge to revert.
        dist: dict[Cell, float] = {c: 0.0 for c in island}
        came_from: dict[Cell, Cell] = {}
        heap = [(0.0, c) for c in island]
        heapq.heapify(heap)
        reached = None
        while heap:
            d, c = heapq.heappop(heap)
            if d > dist.get(c, math.inf):
                continue
            if c in main:
                reached = c
                break
            # Orthogonal steps only: a diagonal's validity depends on its
            # two side cells too (world.strict_diagonal_corners), which
            # this search doesn't track -- an orthogonal step's validity
            # depends only on the destination, so a bridge built from them
            # is always genuinely connected under world.neighbors(),
            # unlike a diagonal one that might look connected here but
            # still get rejected by the corner-cutting rule.
            x, y = c
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if not base.in_bounds(n) or n in base.obstacles:
                    continue
                step = 0.0 if n not in world.obstacles else 1.0
                nd = d + step
                if nd < dist.get(n, math.inf):
                    dist[n] = nd
                    came_from[n] = c
                    heapq.heappush(heap, (nd, n))
        if reached is None:
            continue  # base itself is disconnected here -- nothing to bridge
        c = reached
        while c not in island:
            if c in world.obstacles:
                world.remove_obstacle(c)
            main.add(c)
            c = came_from[c]
        main.update(island)


# Inflated for the single largest robot in the fleet -- used only to decide
# which cells are safe task-point candidates (see _find_clear_approach and
# _push_inward above), so a point offered here works for every robot, not
# just the smallest. Actual route planning still uses each robot's own,
# less conservative, per-radius world from RealWarehouseFleetSim._planning_world_for.
_FLEET_CLEARANCE_WORLD = _inflate_world(
    _WORLD, max(spec["radius"] for spec in ROBOT_SPECS) + EPSILON + 0.3, restore_connectivity=False
)


class TaskState(Enum):
    IDLE = auto()
    TO_PICKUP = auto()
    PICKING = auto()
    TO_DROPOFF = auto()
    DROPPING = auto()


_STATE_LABEL = {
    TaskState.IDLE: "idle",
    TaskState.TO_PICKUP: "→ pickup {task}",
    TaskState.PICKING: "picking up {task}",
    TaskState.TO_DROPOFF: "→ dropoff {task}",
    TaskState.DROPPING: "dropping off {task}",
}


@dataclass
class FloatingText:
    """A short-lived screen callout -- "PICKING UP", a reassignment notice,
    etc -- so state changes that only last a moment (a 0.6s pick/drop pause)
    or that happen once (a task handed to a different robot) are actually
    visible rather than only inferable from the small persistent labels."""
    pos: tuple[float, float]
    text: str
    color: tuple[int, int, int]
    expire_at: float


@dataclass
class RobotAgent:
    robot_id: str
    color: tuple[int, int, int]
    robot: DiffDriveRobot
    cbba: CBBAAgent
    label: str = "AMR"
    state: TaskState = TaskState.IDLE
    task: Task | None = None
    planner: DStarLite | None = None
    path: list[Cell] = field(default_factory=list)
    path_index: int = 0
    pause_timer: float = 0.0
    trail: list[tuple[float, float]] = field(default_factory=list)
    should_yield: bool = False
    spawn_tick: float = 0.0
    stall_timer: float = 0.0
    recovery_timer: float = 0.0
    unreachable_timer: float = 0.0
    congestion_timer: float = 0.0
    detour_active_until: float = 0.0

    def start_route_to(self, world: World, target: Cell) -> None:
        """`world` must be a copy this agent owns (see
        RealWarehouseFleetSim._route_world_for), not a world shared with
        other agents/routes -- start_route_to and refresh_plan both mutate
        it (see the comment below) to keep wherever this robot actually is
        always plannable, and that must never leak into anyone else's plan."""
        start_cell = pos_to_cell(self.robot.position())
        world.remove_obstacle(start_cell)  # a robot standing here proves it's occupiable, see refresh_plan
        self.planner = DStarLite(world, start_cell, target)
        self.path = self.planner.get_path()
        self.path_index = 1 if len(self.path) > 1 else 0

    def refresh_plan(self) -> None:
        cur_cell = pos_to_cell(self.robot.position())
        if self.planner is None:
            return
        if cur_cell in self.planner.world.obstacles:
            # A robot standing on a cell is definitive proof it's really
            # occupiable, whatever the clearance-inflated planning map
            # (built with no idea this specific robot would end up exactly
            # here -- after a pickup/dropoff pause, a detour, anywhere)
            # says about it. Refusing to plan a route FROM a position the
            # robot already, physically holds isn't a safety margin, it's
            # a planner that can never move again. This world is a private
            # per-route copy (see start_route_to), so the exemption stays
            # local to this one robot's current route, not the shared map.
            self.planner.world.remove_obstacle(cur_cell)
            self.planner.notify_obstacles_changed([cur_cell])
        if cur_cell != self.planner.start:
            self.planner.update_start(cur_cell)
            self.planner.compute_shortest_path()
            self.path = self.planner.get_path()
            self.path_index = 1 if len(self.path) > 1 else 0

    def lookahead_target(self) -> Cell:
        while self.path_index < len(self.path) - 1 and dist(
            self.robot.position(), cell_center(self.path[self.path_index])
        ) < ADVANCE_THRESH:
            self.path_index += 1
        if self.path:
            return self.path[self.path_index]
        return self.planner.goal if self.planner else pos_to_cell(self.robot.position())


class RealWarehouseFleetSim:
    def __init__(self) -> None:
        pygame.init()
        self.window_w, self.window_h = INITIAL_WINDOW_W, INITIAL_WINDOW_H
        self.screen = pygame.display.set_mode((self.window_w, self.window_h), pygame.RESIZABLE)
        pygame.display.set_caption("Real Warehouse Fleet Sim - D* Lite + NH-ORCA + CBBA/ED-CBBA + MD-PIBT/Karma")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 15)
        self.small_font = pygame.font.SysFont("consolas", 12)

        # Camera: (cam_x, cam_y) is the world (grid-cell) point shown at the
        # top-left of the viewport (just below the top bar); zoom is pixels
        # per grid cell. Centered on the map at a comfortable zoom level by
        # default -- see world_to_px/px_to_world/cell_rect below, and
        # run()/the event loop for pan (right-drag, arrow keys, WASD) and
        # zoom (mouse wheel) controls.
        self.zoom = DEFAULT_ZOOM
        self.cam_x = GRID_W / 2 - (self.window_w / self.zoom) / 2
        self.cam_y = GRID_H / 2 - ((self.window_h - TOP_BAR) / self.zoom) / 2
        self._panning = False
        self._pan_last: tuple[int, int] = (0, 0)

        self.world = _WORLD
        self.pod_cells = _SHELF_CELLS
        self.wall_cells = _WALL_CELLS

        self.pickup_cells = _shelf_pickup_points(self.world, _SHELF_COMPONENTS)
        self.dropoff_cells = _entrance_dropoff_points(self.world, self.wall_cells)
        if not self.pickup_cells or not self.dropoff_cells:
            raise RuntimeError(
                "Map analysis found no pickup or dropoff points -- "
                f"pickups={len(self.pickup_cells)} dropoffs={len(self.dropoff_cells)}. "
                "Check the map image against environment/mapio.py's wall/shelf classification."
            )

        self.bus = InProcessBus()
        self.karma = KarmaLedger(tau=0.5, payment=1)
        self.resolver = ConflictResolver(karma=self.karma, conflict_radius=CONFLICT_RADIUS, stuck_radius=STUCK_RADIUS)
        self.metrics = FleetMetrics()
        self._inflated_worlds: dict[float, World] = {}

        self.agents: dict[str, RobotAgent] = {}
        spawn_cells = self._pick_spawn_cells(N_ROBOTS)
        for i in range(N_ROBOTS):
            rid = f"r{i}"
            spec = ROBOT_SPECS[i]
            cx, cy = cell_center(spawn_cells[i])
            robot = DiffDriveRobot(rid, Pose(cx, cy, random.uniform(-math.pi, math.pi)),
                                    radius=spec["radius"], max_speed=spec["max_speed"],
                                    max_omega=spec["max_omega"], wheel_base=WHEEL_BASE)
            cbba = CBBAAgent(
                agent_id=rid, bus=self.bus,
                get_position=(lambda r=robot: r.position()), max_bundle=2,
                can_participate=(lambda r=robot: any(
                    self.world.is_free(n) for n in self.world.neighbors(pos_to_cell(r.position()))
                )),
            )
            self.agents[rid] = RobotAgent(robot_id=rid, color=ROBOT_COLORS[i % len(ROBOT_COLORS)],
                                           robot=robot, cbba=cbba, label=spec["label"])

        self.tick = 0
        self.sim_time = 0.0
        self.time_since_task = 0.0
        self.next_task_id = 0
        self.paused = False
        self.was_colliding: set[frozenset] = set()

        self.floating_texts: list[FloatingText] = []
        self.event_log: list[str] = []
        # task_id -> the robot it previously failed on, set by _reannounce
        # and consumed the moment some robot actually starts working it, so
        # a handoff gets called out exactly once, at the point it becomes
        # visible (a new robot moving toward that pickup/dropoff).
        self._task_origin_robot: dict[str, str] = {}

    def _spawn_floating_text(self, pos: tuple[float, float], text: str, color: tuple[int, int, int],
                              duration: float = 1.5) -> None:
        self.floating_texts.append(FloatingText(pos, text, color, self.sim_time + duration))

    def _log_event(self, message: str) -> None:
        self.event_log.append(f"[{self.sim_time:6.1f}s] {message}")
        if len(self.event_log) > 6:
            self.event_log.pop(0)

    # -- camera: pan/zoom viewport ---------------------------------------
    def world_to_px(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.cam_x) * self.zoom), int(TOP_BAR + (y - self.cam_y) * self.zoom)

    def px_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        return self.cam_x + sx / self.zoom, self.cam_y + (sy - TOP_BAR) / self.zoom

    def cell_rect(self, c: Cell) -> pygame.Rect:
        px, py = self.world_to_px(*c)
        size = max(1, round(self.zoom))
        return pygame.Rect(px, py, size, size)

    def _zoom_at(self, screen_pos: tuple[int, int], factor: float) -> None:
        """Zoom in/out by `factor`, keeping the world point currently under
        `screen_pos` fixed on screen (standard scroll-to-cursor behavior)."""
        wx, wy = self.px_to_world(*screen_pos)
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom * factor))
        sx, sy = screen_pos
        self.cam_x = wx - sx / self.zoom
        self.cam_y = wy - (sy - TOP_BAR) / self.zoom

    def _visible_cell_range(self) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) inclusive grid-cell bounds currently on screen,
        clamped to the map -- so _draw only touches cells that are actually
        visible instead of the whole (possibly much bigger than the
        viewport) grid every frame."""
        x0 = max(0, int(self.cam_x) - 1)
        y0 = max(0, int(self.cam_y) - 1)
        x1 = min(GRID_W - 1, int(self.cam_x + self.window_w / self.zoom) + 1)
        y1 = min(GRID_H - 1, int(self.cam_y + (self.window_h - TOP_BAR) / self.zoom) + 1)
        return x0, y0, x1, y1

    def _pick_spawn_cells(self, n: int) -> list[Cell]:
        exclude = set(self.pickup_cells) | set(self.dropoff_cells)
        x0, y0, x1, y1 = _WALL_BBOX
        candidates = [
            c for c in find_clear_cells(self.world, radius=1)
            if c not in exclude and x0 <= c[0] <= x1 and y0 <= c[1] <= y1
        ]
        random.shuffle(candidates)
        chosen: list[Cell] = []
        min_sep = max(3.0, max((spec["radius"] for spec in ROBOT_SPECS), default=1.0) * 4)
        for c in candidates:
            if all(dist(cell_center(c), cell_center(o)) >= min_sep for o in chosen):
                chosen.append(c)
            if len(chosen) >= n:
                break
        if len(chosen) < n:
            # Sparse map or overly strict separation -- fill the rest from
            # whatever's left rather than crashing on too few spawn points.
            for c in candidates:
                if c not in chosen:
                    chosen.append(c)
                if len(chosen) >= n:
                    break
        return chosen

    def _planning_world_for(self, radius: float) -> World:
        """The clearance-inflated world (see _inflate_world) D* Lite should
        plan against for a robot of this radius. Cached per distinct radius
        -- ROBOT_SPECS only has two -- and computed lazily so a fresh sim
        doesn't pay the inflation cost for a radius nothing ends up using."""
        key = round(radius, 3)
        cached = self._inflated_worlds.get(key)
        if cached is None:
            cached = _inflate_world(self.world, radius + EPSILON + 0.3)
            # Pickup/dropoff points sit immediately next to a shelf or wall
            # by design (that's the point of a docking location), so
            # inflation can legitimately swallow the goal cell itself --
            # D* Lite would then see an unreachable goal (DStarLite seeds
            # rhs[goal]=0 unconditionally, but every edge INTO a non-free
            # goal costs INF, so no path ever closes the last step). These
            # cells are already validated task points, so keep them free
            # regardless of what inflation says about their surroundings.
            for c in self.pickup_cells + self.dropoff_cells:
                cached.remove_obstacle(c)
            self._inflated_worlds[key] = cached
        return cached

    def _route_world_for(self, agent: RobotAgent) -> World:
        """A fresh, private copy of the shared per-radius planning world for
        one agent's new route. start_route_to/refresh_plan mutate this copy
        (clearing whatever cell the robot is actually standing on, even if
        the static map calls it obstacle) as the route progresses -- must
        never be the shared cached instance from _planning_world_for, or
        one robot's current position would leak into every other robot's
        plan on the same cache entry."""
        shared = self._planning_world_for(agent.robot.radius)
        route_world = World(shared.width, shared.height, strict_diagonal_corners=shared.strict_diagonal_corners)
        route_world.obstacles = set(shared.obstacles)
        return route_world

    # -- task lifecycle -------------------------------------------------
    def _maybe_spawn_task(self, dt: float) -> None:
        self.time_since_task += dt
        registry = _first_agent(self.agents)
        unassigned = [tid for tid in registry.tasks if registry.winning_agents.get(tid) is None]
        if self.time_since_task >= TASK_SPAWN_INTERVAL_S and len(unassigned) < MAX_PENDING_TASKS:
            self.time_since_task = 0.0
            pickup = random.choice(self.pickup_cells)
            dropoff = random.choice(self.dropoff_cells)
            task = Task(task_id=f"task{self.next_task_id}", pickup=pickup, dropoff=dropoff,
                        reward=TASK_REWARD, created_tick=self.tick)
            self.next_task_id += 1
            announcer = random.choice(list(self.agents.values()))
            announcer.cbba.announce_task(task)

    # -- main loop --------------------------------------------------------
    def run(self) -> None:
        running = True
        while running:
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.window_w, self.window_h = event.w, event.h
                    self.screen = pygame.display.set_mode((self.window_w, self.window_h), pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    self._panning = True
                    self._pan_last = event.pos
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                    self._panning = False
                elif event.type == pygame.MOUSEMOTION and self._panning:
                    dx, dy = event.pos[0] - self._pan_last[0], event.pos[1] - self._pan_last[1]
                    self.cam_x -= dx / self.zoom
                    self.cam_y -= dy / self.zoom
                    self._pan_last = event.pos
                elif event.type == pygame.MOUSEWHEEL:
                    self._zoom_at(pygame.mouse.get_pos(), ZOOM_STEP if event.y > 0 else 1.0 / ZOOM_STEP)

            keys = pygame.key.get_pressed()
            pan = PAN_SPEED_CELLS_PER_S * DT
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.cam_x -= pan
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.cam_x += pan
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                self.cam_y -= pan
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                self.cam_y += pan

            if not self.paused:
                self._step(DT)

            self._draw()
        pygame.quit()
        sys.exit(0)

    def _handle_click(self, mouse_pos) -> None:
        x, y = mouse_pos
        if y < TOP_BAR:
            return
        wx, wy = self.px_to_world(x, y)
        cell = (int(math.floor(wx)), int(math.floor(wy)))
        if not self.world.in_bounds(cell):
            return
        if cell in self.pickup_cells or cell in self.dropoff_cells:
            return
        if cell in self.pod_cells or cell in self.wall_cells:
            return  # permanent warehouse structure, not a droppable obstacle
        occupied = any(pos_to_cell(a.robot.position()) == cell for a in self.agents.values())
        if occupied:
            return
        changed = self.world.toggle_obstacle(cell)
        if changed:
            # Mirror the SAME direction of change (not another blind
            # toggle_obstacle -- a planning world can already have this
            # exact cell marked obstacle for an unrelated reason, clearance
            # inflation near some OTHER obstacle, so re-toggling it there
            # could flip it the wrong way) into every cached planning world
            # and every agent's own live planner world.
            added = cell in self.world.obstacles
            for planning_world in self._inflated_worlds.values():
                (planning_world.add_obstacle if added else planning_world.remove_obstacle)(cell)
            for agent in self.agents.values():
                if agent.planner is not None:
                    # A robot already mid-route is navigating its own
                    # PRIVATE copy of the planning world (see
                    # _route_world_for/RobotAgent.start_route_to -- needed
                    # so one robot's current-cell exemption never leaks
                    # into another robot's plan), not the shared cache
                    # mutated above. Updating only the shared cache would
                    # leave that copy, and therefore this robot's actual
                    # route, completely unaware a new obstacle exists --
                    # notify_obstacles_changed has nothing to propagate if
                    # the cell was never added to *this* world in the
                    # first place. Apply it here too before notifying.
                    (agent.planner.world.add_obstacle if added else agent.planner.world.remove_obstacle)(cell)
                    agent.planner.notify_obstacles_changed([cell])
                    agent.path = agent.planner.get_path()
                    agent.path_index = min(agent.path_index, max(len(agent.path) - 1, 0))

    def _step(self, dt: float) -> None:
        self.tick += 1
        self.sim_time += dt

        self._maybe_spawn_task(dt)
        for agent in self.agents.values():
            agent.cbba.step_tick()

        for agent in self.agents.values():
            self._step_agent_task_fsm(agent, dt)

        views = {
            rid: RobotView(rid, agent.robot.position(), agent.robot.velocity_at_reference_point(0.0))
            for rid, agent in self.agents.items() if agent.state != TaskState.IDLE
        }
        should_yield = self.resolver.resolve(views, self.tick) if views else {}
        for agent in self.agents.values():
            agent.should_yield = should_yield.get(agent.robot_id, False)

        neighbors = [a.robot for a in self.agents.values()]
        for agent in self.agents.values():
            self._step_agent_motion(agent, neighbors, dt)

        self._check_collisions()

    def _step_agent_task_fsm(self, agent: RobotAgent, dt: float) -> None:
        if agent.state == TaskState.IDLE:
            agent.robot.set_body_velocity(0.0, 0.0)
            next_task = agent.cbba.next_task()
            if next_task is not None:
                cur_cell = pos_to_cell(agent.robot.position())
                if any(self.world.is_free(n) for n in self.world.neighbors(cur_cell)):
                    agent.task = next_task
                    agent.state = TaskState.TO_PICKUP
                    agent.spawn_tick = self.sim_time
                    agent.start_route_to(self._route_world_for(agent), next_task.pickup)
                    origin = self._task_origin_robot.pop(next_task.task_id, None)
                    if origin is not None:
                        self._log_event(f"{agent.robot_id} picked up {next_task.task_id} (reassigned from {origin})")
                        self._spawn_floating_text(
                            agent.robot.position(), f"TOOK OVER FROM {origin.upper()}", COLOR_WARN, duration=2.5
                        )
                else:
                    # CBBA's distance-only scoring doesn't know this robot is
                    # currently boxed in with nowhere to go -- don't waste
                    # the task on it, hand it straight to the fleet again.
                    self._reannounce(next_task, exclude=agent, reason="boxed in, no free neighbor")
        elif agent.state == TaskState.PICKING:
            agent.robot.set_body_velocity(0.0, 0.0)
            agent.pause_timer -= dt
            if agent.pause_timer <= 0:
                agent.state = TaskState.TO_DROPOFF
                agent.start_route_to(self._route_world_for(agent), agent.task.dropoff)
        elif agent.state == TaskState.DROPPING:
            agent.robot.set_body_velocity(0.0, 0.0)
            agent.pause_timer -= dt
            if agent.pause_timer <= 0:
                self.metrics.task_completed(self.sim_time - agent.spawn_tick)
                self._log_event(f"{agent.robot_id} completed {agent.task.task_id}")
                agent.cbba.release_task(agent.task.task_id)
                agent.task = None
                agent.planner = None
                agent.path = []
                agent.state = TaskState.IDLE
        # TO_PICKUP / TO_DROPOFF handled in _step_agent_motion (needs neighbor list)

    def _step_agent_motion(self, agent: RobotAgent, neighbors, dt: float) -> None:
        if agent.state not in (TaskState.TO_PICKUP, TaskState.TO_DROPOFF):
            return
        detouring = self.sim_time < agent.detour_active_until
        if not detouring:
            agent.refresh_plan()
        target_cell = agent.planner.goal

        reachable = bool(agent.path) and agent.path[-1] == target_cell
        if not reachable:
            agent.unreachable_timer += dt
            agent.robot.set_body_velocity(0.0, 0.0)
            if agent.unreachable_timer >= UNREACHABLE_REASSIGN_S:
                self._reassign_stuck_task(agent)
            return
        agent.unreachable_timer = 0.0

        lookahead = agent.lookahead_target()

        ref_pos = agent.robot.reference_point(EPSILON)
        pref = toward(ref_pos, cell_center(lookahead), agent.robot.max_speed)
        if agent.should_yield:
            pref = (pref[0] * 0.05, pref[1] * 0.05)

        if agent.recovery_timer > 0:
            agent.recovery_timer -= dt
            desired_heading = math.atan2(pref[1], pref[0]) if math.hypot(*pref) > 1e-6 else agent.robot.pose.theta
            recovery_omega = max(-agent.robot.max_omega, min(
                agent.robot.max_omega, RECOVERY_GAIN * angle_diff(desired_heading, agent.robot.pose.theta)
            ))
            agent.robot.set_body_velocity(0.0, recovery_omega)
        else:
            obstacles = self._nearby_static_obstacles(agent.robot.position())
            v, w = nh_orca_velocity(agent.robot, neighbors, pref, TIME_HORIZON, dt, epsilon=EPSILON,
                                     static_obstacles=obstacles, static_time_horizon=NH_ORCA_STATIC_TIME_HORIZON)
            agent.robot.set_body_velocity(v, w)

            if abs(v) < STALL_SPEED_THRESH and not agent.should_yield:
                agent.stall_timer += dt
            else:
                agent.stall_timer = 0.0
            if agent.stall_timer > STALL_THRESHOLD_S:
                agent.recovery_timer = RECOVERY_BURST_S
                agent.stall_timer = 0.0

        if not detouring:
            if abs(agent.robot.velocity[0]) < JAM_SPEED_THRESH:
                agent.congestion_timer += dt
            else:
                agent.congestion_timer = 0.0
            if agent.congestion_timer >= CONGESTION_TIMEOUT_S:
                self._attempt_congestion_detour(agent)
                agent.congestion_timer = 0.0

        agent.robot.step(dt)
        agent.trail.append(agent.robot.position())
        if len(agent.trail) > 200:
            agent.trail.pop(0)

        if dist(agent.robot.position(), cell_center(target_cell)) < ARRIVE_THRESH:
            agent.pause_timer = PICK_DROP_SECONDS
            agent.state = TaskState.PICKING if agent.state == TaskState.TO_PICKUP else TaskState.DROPPING
            agent.robot.set_body_velocity(0.0, 0.0)
            verb = "PICKING UP" if agent.state == TaskState.PICKING else "DROPPING OFF"
            self._spawn_floating_text(agent.robot.position(), verb, agent.color, duration=PICK_DROP_SECONDS + 0.6)

    def _attempt_congestion_detour(self, agent: RobotAgent) -> bool:
        cur_cell = pos_to_cell(agent.robot.position())
        goal = agent.planner.goal
        phantom_obstacles: set[Cell] = set()
        for other_agent in self.agents.values():
            other = other_agent.robot
            if other is agent.robot:
                continue
            if dist(other.position(), agent.robot.position()) >= JAM_RADIUS:
                continue
            if abs(other.velocity[0]) >= JAM_SPEED_THRESH:
                continue
            cell = pos_to_cell(other.position())
            if cell not in (cur_cell, goal):
                phantom_obstacles.add(cell)

        upcoming = agent.path[agent.path_index:agent.path_index + 2]
        for cell in upcoming:
            if cell not in (cur_cell, goal):
                phantom_obstacles.add(cell)

        if not phantom_obstacles:
            return False

        # Base on the agent's own live planning world, not the shared cache
        # -- it may already have this agent's current cell exempted (see
        # RobotAgent.refresh_plan), which the shared copy never is.
        scratch = World(self.world.width, self.world.height, strict_diagonal_corners=True)
        scratch.obstacles = set(agent.planner.world.obstacles) | phantom_obstacles
        if not scratch.is_free(cur_cell) or not scratch.is_free(goal):
            return False

        detour_path = DStarLite(scratch, cur_cell, goal).get_path()
        if not detour_path or detour_path[-1] != goal:
            return False

        original_remaining = agent.path[agent.path_index:] or [cur_cell]
        original_cost = _path_cost([cur_cell] + original_remaining)
        detour_cost = _path_cost(detour_path)
        if detour_cost > original_cost * DETOUR_MAX_COST_RATIO:
            return False

        agent.path = detour_path
        agent.path_index = 1 if len(detour_path) > 1 else 0
        agent.detour_active_until = self.sim_time + DETOUR_COMMIT_S
        return True

    def _reassign_stuck_task(self, agent: RobotAgent) -> None:
        old_task = agent.task
        agent.task = None
        agent.planner = None
        agent.path = []
        agent.unreachable_timer = 0.0
        agent.state = TaskState.IDLE
        self._reannounce(old_task, exclude=agent, reason="no route to target")

    def _reannounce(self, task: Task, exclude: RobotAgent, reason: str) -> None:
        """Release `task` from `exclude` and re-announce it under a fresh
        id. Records `exclude` as the task's origin robot (see
        _task_origin_robot) so the fleet member that eventually wins the
        retry can be called out as taking over a failed task, visibly --
        the whole point users asked for here: a silent CBBA re-bid looks
        identical to a routine new task otherwise."""
        exclude.cbba.release_task(task.task_id)
        retry = Task(task_id=f"{task.task_id}-retry{self.next_task_id}",
                     pickup=task.pickup, dropoff=task.dropoff,
                     reward=task.reward, created_tick=self.tick)
        self.next_task_id += 1
        self._task_origin_robot[retry.task_id] = exclude.robot_id
        self._log_event(f"{exclude.robot_id} failed {task.task_id} ({reason}) -- reannounced as {retry.task_id}")
        self._spawn_floating_text(exclude.robot.position(), "TASK FAILED - REASSIGNING", COLOR_WARN, duration=2.0)
        candidates = [a for a in self.agents.values() if a is not exclude]
        announcer = random.choice(candidates) if candidates else exclude
        announcer.cbba.announce_task(retry)

    _OBSTACLE_SEARCH_RADIUS = sm(1.2)
    _OBSTACLE_POINT_RADIUS = sm(0.02)
    _MAX_OBSTACLE_POINTS = 6

    def _nearby_static_obstacles(self, pos: tuple[float, float]) -> list[tuple[tuple[float, float], float]]:
        x, y = pos
        r = self._OBSTACLE_SEARCH_RADIUS
        x0, x1 = int(math.floor(x - r)), int(math.ceil(x + r))
        y0, y1 = int(math.floor(y - r)), int(math.ceil(y + r))
        found: list[tuple[float, tuple[float, float]]] = []
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                if (cx, cy) not in self.world.obstacles:
                    continue
                nx = max(cx, min(x, cx + 1.0))
                ny = max(cy, min(y, cy + 1.0))
                d = dist((x, y), (nx, ny))
                if d < r:
                    found.append((d, (nx, ny)))
        found.sort(key=lambda t: t[0])
        return [(pt, self._OBSTACLE_POINT_RADIUS) for _, pt in found[: self._MAX_OBSTACLE_POINTS]]

    def _check_collisions(self) -> None:
        ids = list(self.agents.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = self.agents[ids[i]], self.agents[ids[j]]
                d = dist(a.robot.position(), b.robot.position())
                key = frozenset((ids[i], ids[j]))
                colliding = d < (a.robot.radius + b.robot.radius)
                if colliding and key not in self.was_colliding:
                    self.metrics.register_collision()
                if colliding:
                    self.was_colliding.add(key)
                else:
                    self.was_colliding.discard(key)

    # -- rendering --------------------------------------------------------
    def _draw(self) -> None:
        screen = self.screen
        screen.fill(COLOR_BG)

        x0, y0, x1, y1 = self._visible_cell_range()
        draw_grid_lines = self.zoom >= 4  # too fine to matter (and to see) once zoomed way out
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                c = (x, y)
                rect = self.cell_rect(c)
                if c in self.pod_cells:
                    pygame.draw.rect(screen, COLOR_POD, rect)
                    pygame.draw.rect(screen, COLOR_POD_EDGE, rect, 1)
                    continue
                if c in self.world.obstacles:
                    pygame.draw.rect(screen, COLOR_OBSTACLE, rect)
                else:
                    pygame.draw.rect(screen, COLOR_FREE, rect)
                if draw_grid_lines:
                    pygame.draw.rect(screen, COLOR_GRID_LINE, rect, 1)

        for c in self.pickup_cells:
            pygame.draw.rect(screen, COLOR_PICKUP, self.cell_rect(c), 2)
        for c in self.dropoff_cells:
            pygame.draw.rect(screen, COLOR_DROPOFF, self.cell_rect(c), 2)

        for agent in self.agents.values():
            if len(agent.trail) > 1:
                pts = [self.world_to_px(x, y) for x, y in agent.trail]
                pygame.draw.lines(screen, agent.color, False, pts, 1)
            if agent.path and agent.state in (TaskState.TO_PICKUP, TaskState.TO_DROPOFF):
                for c in agent.path:
                    if c not in self.world.obstacles:
                        r = self.cell_rect(c).inflate(-self.zoom * 0.7, -self.zoom * 0.7)
                        pygame.draw.rect(screen, agent.color, r, 1)

        # Active-target callout: a straight dashed line from robot to its
        # task's ultimate goal (pickup while inbound/docked, dropoff once
        # carrying), plus a pulsing ring around that cell in the robot's
        # own color -- an at-a-glance "this one's headed there", distinct
        # from the fine-grained D* Lite path outline above.
        pulse = 2 + round(2 * (0.5 + 0.5 * math.sin(self.sim_time * 4)))
        for agent in self.agents.values():
            target = _active_target(agent)
            if target is None:
                continue
            p1 = self.world_to_px(*agent.robot.position())
            p2 = self.world_to_px(*cell_center(target))
            _draw_dashed_line(screen, p1, p2, agent.color, dash_len=8)
            pygame.draw.rect(screen, agent.color, self.cell_rect(target).inflate(pulse * 2, pulse * 2), 2)

        edge_positions = {rid: a.robot.position() for rid, a in self.agents.items()}
        for yielder, winner in self.resolver.dependency_edges:
            if yielder in edge_positions and winner in edge_positions:
                p1 = self.world_to_px(*edge_positions[yielder])
                p2 = self.world_to_px(*edge_positions[winner])
                pygame.draw.line(screen, COLOR_YIELD_EDGE, p1, p2, 2)

        for agent in self.agents.values():
            cx, cy = self.world_to_px(agent.robot.pose.x, agent.robot.pose.y)
            r_px = max(2, int(agent.robot.radius * self.zoom))
            fill_color = agent.color if not agent.should_yield else tuple(c // 2 for c in agent.color)
            pygame.draw.circle(screen, fill_color, (cx, cy), r_px)
            pygame.draw.circle(screen, (10, 10, 12), (cx, cy), r_px, 2)

            carrying = agent.state in (TaskState.TO_DROPOFF, TaskState.DROPPING)
            if carrying:
                box = max(6, int(r_px * 0.9))
                box_rect = pygame.Rect(0, 0, box, box)
                box_rect.center = (cx, cy)
                pygame.draw.rect(screen, COLOR_CARGO, box_rect)
                pygame.draw.rect(screen, COLOR_CARGO_EDGE, box_rect, 2)
                pygame.draw.line(screen, COLOR_CARGO_EDGE, box_rect.midtop, box_rect.midbottom, 1)
                pygame.draw.line(screen, COLOR_CARGO_EDGE, box_rect.midleft, box_rect.midright, 1)
            else:
                hx = cx + int(math.cos(agent.robot.pose.theta) * r_px)
                hy = cy + int(math.sin(agent.robot.pose.theta) * r_px)
                pygame.draw.line(screen, (10, 10, 12), (cx, cy), (hx, hy), 2)

            if agent.unreachable_timer > 0:
                pygame.draw.circle(screen, COLOR_WARN, (cx, cy), r_px + 5, 2)
                if agent.planner is not None:
                    tx, ty = self.world_to_px(*cell_center(agent.planner.goal))
                    _draw_dashed_line(screen, (cx, cy), (tx, ty), COLOR_WARN)
                    pygame.draw.circle(screen, COLOR_WARN, (tx, ty), 5, 1)
                tag = "NO ROUTE TO TARGET" if agent.unreachable_timer < UNREACHABLE_REASSIGN_S else "REASSIGNING"
                warn_label = self.small_font.render(tag, True, COLOR_WARN)
                screen.blit(warn_label, (cx - warn_label.get_width() // 2, cy + r_px + 4))

            header = self.small_font.render(
                f"{agent.robot_id}:{agent.label} k={self.karma.balance(agent.robot_id)}", True, COLOR_TEXT
            )
            task_id = agent.task.task_id if agent.task else ""
            action = self.small_font.render(_STATE_LABEL[agent.state].format(task=task_id), True, COLOR_DIM)
            screen.blit(header, (cx - header.get_width() // 2, cy - r_px - 30))
            screen.blit(action, (cx - action.get_width() // 2, cy - r_px - 16))

        self._draw_floating_texts()
        self._draw_top_bar()
        self._draw_event_log()
        pygame.display.flip()

    def _draw_floating_texts(self) -> None:
        self.floating_texts = [ft for ft in self.floating_texts if ft.expire_at > self.sim_time]
        for ft in self.floating_texts:
            fade_window = 0.4
            alpha = 255 if ft.expire_at - self.sim_time > fade_window else int(
                255 * max(0.0, (ft.expire_at - self.sim_time) / fade_window)
            )
            px, py = self.world_to_px(*ft.pos)
            surf = self.small_font.render(ft.text, True, ft.color)
            surf.set_alpha(alpha)
            self.screen.blit(surf, (px - surf.get_width() // 2, py - 28))

    def _draw_event_log(self) -> None:
        """Recent-events panel (task completions, failures, reassignment
        handoffs) -- a running commentary so a handoff between robots is
        readable even if you miss the floating callout at the moment it
        happens."""
        if not self.event_log:
            return
        pad, line_h = 6, 16
        width = 460
        height = pad * 2 + line_h * len(self.event_log)
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((10, 11, 14, 210))
        for i, message in enumerate(self.event_log):
            panel.blit(self.small_font.render(message, True, COLOR_TEXT), (pad, pad + i * line_h))
        self.screen.blit(panel, (10, self.window_h - height - 10))

    def _draw_top_bar(self) -> None:
        screen = self.screen
        pygame.draw.rect(screen, (10, 11, 14), (0, 0, self.window_w, TOP_BAR))

        total_ed_msgs = sum(a.cbba.messages_sent for a in self.agents.values())
        total_periodic_msgs = sum(a.cbba.periodic_message_cost for a in self.agents.values())
        reduction = (1 - total_ed_msgs / total_periodic_msgs) * 100 if total_periodic_msgs else 0.0
        total_expansions = sum(a.planner.node_expansions for a in self.agents.values() if a.planner)

        bundles = " ".join(
            f"{rid}:[{','.join(a.cbba.bundle) or '-'}]" for rid, a in self.agents.items()
        )

        lines = [
            "Click: obstacle   Right-drag/Arrows/WASD: pan   Wheel: zoom   SPACE: pause   ESC: quit",
            f"Tasks completed: {self.metrics.tasks_completed}   mean duration: {self.metrics.mean_task_duration():.1f}s"
            f"   |   collisions: {self.metrics.collision_count}"
            f"   |   D* Lite cumulative expansions: {total_expansions}",
            f"CBBA messages -- ED-CBBA: {total_ed_msgs}   periodic-equivalent: {total_periodic_msgs}"
            f"   (reduction: {reduction:.0f}%)   |   Karma variance: {self.karma.variance():.2f}"
            f"   |   active deadlocks: {len(self.resolver._deadlock_start_tick)}",
            f"Bundles: {bundles}",
        ]
        for i, line in enumerate(lines):
            color = COLOR_WARN if (i == 1 and self.metrics.collision_count > 0) else COLOR_TEXT
            font = self.font if i == 0 else self.small_font
            screen.blit(font.render(line, True, color), (10, 4 + i * 20))


def _first_agent(agents: dict[str, RobotAgent]) -> CBBAAgent:
    return next(iter(agents.values())).cbba


def main() -> None:
    RealWarehouseFleetSim().run()


if __name__ == "__main__":
    main()
