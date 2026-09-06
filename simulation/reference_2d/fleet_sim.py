"""Integrated fleet simulator: every layer running together in one window.

D* Lite (per-robot global planning) + NH-ORCA (local avoidance) +
CBBA/ED-CBBA (decentralized task bidding) + Karma-weighted priority
conflict resolution, all sharing one grid/world and one in-process message
bus. Click a free cell to drop an obstacle mid-run -- every robot with an
active plan replans incrementally, and if that reroute pushes two robots
into each other's way, watch the dependency-graph arrow appear as one
yields to the other.

Run: python -m simulation.reference_2d.fleet_sim
"""
from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from enum import Enum, auto

import pygame

from algorithms.task_allocation.cbba import CBBAAgent, Task
from algorithms.local_planning.nh_orca import DEFAULT_EPSILON, nh_orca_velocity
from communication.inprocess import InProcessBus
from algorithms.conflict_resolution.karma import KarmaLedger
from algorithms.conflict_resolution.mdpibt import ConflictResolver, RobotView
from environment.metrics import FleetMetrics
from algorithms.global_planning.dstar_lite import DStarLite
from models.robot import DiffDriveRobot, Pose
from environment.grid_world import Cell, World

CELL_PX = 26
GRID_W, GRID_H = 30, 18
TOP_BAR = 110
WINDOW_W, WINDOW_H = GRID_W * CELL_PX, GRID_H * CELL_PX + TOP_BAR

# Amazon-style pod grid: 2x2 storage pods separated by 2-cell-wide aisles on
# every side, so a robot approaching from any of the 4 directions always has
# a full aisle to work with (the real Kiva/Amazon-Robotics layout). The zone
# is a clean multiple of the 4-cell period so pods tile with no partial pod
# clipped at an edge.
POD_ZONE = (3, 2, 27, 14)  # (x0, y0, x1, y1) exclusive
POD_SIZE = 2
POD_PERIOD = 4

DT = 1.0 / 60.0
TIME_HORIZON = 2.0
ARRIVE_THRESH = 0.22
ADVANCE_THRESH = 0.45
PICK_DROP_SECONDS = 0.6
TASK_SPAWN_INTERVAL_S = 3.0
MAX_PENDING_TASKS = 4

# Recovery for a specific NH-ORCA failure mode: the epsilon reference-point
# inversion divides by epsilon, so when a robot's heading has drifted far
# from its direction of travel *and* ORCA is heavily constraining it (tight
# proximity to another robot), the safe velocity it finds can invert to only
# a tiny corrective spin -- never enough to realign, so it never escapes.
# A short burst of pure in-place rotation (zero translation, so it can't
# cause a new collision) breaks the loop by fixing the heading directly.
STALL_SPEED_THRESH = 0.05
STALL_THRESHOLD_S = 1.0
RECOVERY_BURST_S = 0.4
RECOVERY_GAIN = 4.0

# If a robot's D* Lite goal is genuinely unreachable (e.g. the user has
# boxed it in with obstacles), waiting forever helps no one -- after this
# long with no path, abandon the task and re-announce it fresh so another
# robot gets a chance to bid on it instead.
UNREACHABLE_REASSIGN_S = 3.0

# Congestion detour: D* Lite only ever reasons about static obstacles, so a
# multi-robot pileup has no way to become "go a different way" on its own --
# NH-ORCA/Karma can only ever produce a slow-down-and-take-turns decision in
# the SAME spot. When a robot has made near-zero progress for this long with
# another near-stationary robot nearby, treat nearby stuck robots' current
# cells as temporary obstacles and compute a real detour around them, rather
# than continuing to negotiate forever over the same contested cell.
JAM_SPEED_THRESH = 0.05
JAM_RADIUS = 1.2
CONGESTION_TIMEOUT_S = 1.2
DETOUR_COMMIT_S = 3.0
# A detour must cost no more than this multiple of the route it's replacing
# -- otherwise blocking just the immediate next cell can make "backtrack the
# whole way around" look like the only option left in a scratch world, when
# the real fix is just patience (or a genuinely different nearby route).
DETOUR_MAX_COST_RATIO = 1.6

# NH-ORCA's static-obstacle lines used the same long time horizon as
# robot-robot avoidance (TIME_HORIZON). ORCA's velocity obstacle is a
# *linear* extrapolation of current velocity: a long horizon is right for
# another robot (react to its true motion early) but overstates a static
# corner's threat when the robot's actual path curves around it -- worse
# the larger and slower the robot, since the extrapolated cone scales with
# both. That's what stalled the HEAVY robot at a row-transition next to a
# freshly-narrowed aisle: three nearby static points (an added obstacle
# plus the next pod row) combined into a heavily restrictive constraint
# under a 2s horizon even though it was never actually close to colliding.
NH_ORCA_STATIC_TIME_HORIZON = 0.4

# One heavy-lift AMR (much larger, slower, harder to turn) alongside three
# standard drive units -- a small heterogeneous-fleet touch real Amazon
# warehouses have (pallet movers vs. standard pod carriers).
ROBOT_SPECS = [
    {"radius": 0.42, "max_speed": 0.85, "max_omega": 2.0, "label": "HEAVY"},
    {"radius": 0.28, "max_speed": 1.15, "max_omega": 3.2, "label": "AMR"},
    {"radius": 0.28, "max_speed": 1.15, "max_omega": 3.2, "label": "AMR"},
    {"radius": 0.28, "max_speed": 1.15, "max_omega": 3.2, "label": "AMR"},
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


def world_to_px(x: float, y: float) -> tuple[int, int]:
    return int(x * CELL_PX), int(TOP_BAR + y * CELL_PX)


def cell_rect(c: Cell) -> pygame.Rect:
    x, y = c
    return pygame.Rect(x * CELL_PX, TOP_BAR + y * CELL_PX, CELL_PX, CELL_PX)


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


def make_pods(world: World) -> set[Cell]:
    """Lay out the storage-pod grid and return the set of pod cells (so the
    renderer/click-handler can tell "warehouse structure" apart from
    obstacles the user drops in later)."""
    x0, y0, x1, y1 = POD_ZONE
    pods: set[Cell] = set()
    for x in range(x0, x1):
        if (x - x0) % POD_PERIOD >= POD_SIZE:
            continue
        for y in range(y0, y1):
            if (y - y0) % POD_PERIOD >= POD_SIZE:
                continue
            pods.add((x, y))
            world.add_obstacle((x, y))
    return pods


class TaskState(Enum):
    IDLE = auto()
    TO_PICKUP = auto()
    PICKING = auto()
    TO_DROPOFF = auto()
    DROPPING = auto()


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
        start_cell = pos_to_cell(self.robot.position())
        self.planner = DStarLite(world, start_cell, target)
        self.path = self.planner.get_path()
        self.path_index = 1 if len(self.path) > 1 else 0

    def refresh_plan(self) -> None:
        cur_cell = pos_to_cell(self.robot.position())
        if self.planner is None:
            return
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


class FleetSim:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Warehouse Fleet Sim - D* Lite + NH-ORCA + CBBA/ED-CBBA + MD-PIBT/Karma")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 15)
        self.small_font = pygame.font.SysFont("consolas", 12)

        self.world = World(GRID_W, GRID_H, strict_diagonal_corners=True)
        self.pod_cells = make_pods(self.world)

        self.pickup_cells = [(1, y) for y in (2, 6, 10, 14)]
        self.dropoff_cells = [(GRID_W - 2, y) for y in (2, 6, 10, 14)]
        for c in self.pickup_cells + self.dropoff_cells:
            self.world.obstacles.discard(c)

        self.bus = InProcessBus()
        self.karma = KarmaLedger(tau=0.5, payment=1)
        self.resolver = ConflictResolver(karma=self.karma, conflict_radius=1.4)
        self.metrics = FleetMetrics()

        self.agents: dict[str, RobotAgent] = {}
        spawn_cells = self._pick_spawn_cells(N_ROBOTS)
        for i in range(N_ROBOTS):
            rid = f"r{i}"
            spec = ROBOT_SPECS[i]
            cx, cy = cell_center(spawn_cells[i])
            robot = DiffDriveRobot(rid, Pose(cx, cy, random.uniform(-math.pi, math.pi)),
                                    radius=spec["radius"], max_speed=spec["max_speed"], max_omega=spec["max_omega"])
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

    def _pick_spawn_cells(self, n: int) -> list[Cell]:
        # Staging/charging area in the open corridor below the pod grid --
        # plenty of clearance on all sides, unlike the narrow perimeter lanes.
        x0, y0, x1, y1 = POD_ZONE
        candidates = [c for c in [(x, y) for x in range(x0 + 1, x1 - 1) for y in (y1 + 1, y1 + 2)]
                      if self.world.is_free(c)]
        random.shuffle(candidates)
        return candidates[:n]

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
                        reward=100.0, created_tick=self.tick)
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
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

            if not self.paused:
                self._step(DT)

            self._draw()
        pygame.quit()
        sys.exit(0)

    def _handle_click(self, mouse_pos) -> None:
        x, y = mouse_pos
        if y < TOP_BAR:
            return
        cell = (x // CELL_PX, (y - TOP_BAR) // CELL_PX)
        if not self.world.in_bounds(cell):
            return
        if cell in self.pickup_cells or cell in self.dropoff_cells:
            return
        if cell in self.pod_cells:
            return  # permanent warehouse structure, not a droppable obstacle
        occupied = any(pos_to_cell(a.robot.position()) == cell for a in self.agents.values())
        if occupied:
            return
        changed = self.world.toggle_obstacle(cell)
        if changed:
            for agent in self.agents.values():
                if agent.planner is not None:
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
                    agent.start_route_to(self.world, next_task.pickup)
                else:
                    # CBBA's distance-only scoring doesn't know this robot is
                    # currently boxed in with nowhere to go -- don't waste
                    # the task on it, hand it straight to the fleet again.
                    self._reannounce(next_task, exclude=agent)
        elif agent.state == TaskState.PICKING:
            agent.robot.set_body_velocity(0.0, 0.0)
            agent.pause_timer -= dt
            if agent.pause_timer <= 0:
                agent.state = TaskState.TO_DROPOFF
                agent.start_route_to(self.world, agent.task.dropoff)
        elif agent.state == TaskState.DROPPING:
            agent.robot.set_body_velocity(0.0, 0.0)
            agent.pause_timer -= dt
            if agent.pause_timer <= 0:
                self.metrics.task_completed(self.sim_time - agent.spawn_tick)
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
            # While a congestion detour is active, agent.path is a one-off
            # route around the jam that the live planner doesn't know about
            # (it only ever sees static obstacles) -- refreshing from it here
            # would immediately discard the detour and steer back into
            # the same jam.
            agent.refresh_plan()
        target_cell = agent.planner.goal

        reachable = bool(agent.path) and agent.path[-1] == target_cell
        if not reachable:
            # Genuinely no path (e.g. boxed in by obstacles) -- waiting
            # forever helps no one. Give it a grace period in case the
            # obstruction clears, then hand the task to another robot.
            agent.unreachable_timer += dt
            agent.robot.set_body_velocity(0.0, 0.0)
            if agent.unreachable_timer >= UNREACHABLE_REASSIGN_S:
                self._reassign_stuck_task(agent)
            return
        agent.unreachable_timer = 0.0

        lookahead = agent.lookahead_target()

        ref_pos = agent.robot.reference_point(DEFAULT_EPSILON)
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
            v, w = nh_orca_velocity(agent.robot, neighbors, pref, TIME_HORIZON, dt, static_obstacles=obstacles,
                                     static_time_horizon=NH_ORCA_STATIC_TIME_HORIZON)
            agent.robot.set_body_velocity(v, w)

            if abs(v) < STALL_SPEED_THRESH and not agent.should_yield:
                agent.stall_timer += dt
            else:
                agent.stall_timer = 0.0
            if agent.stall_timer > STALL_THRESHOLD_S:
                agent.recovery_timer = RECOVERY_BURST_S
                agent.stall_timer = 0.0

        # Congestion detection runs regardless of whether this tick's
        # velocity came from normal NH-ORCA driving or a heading-recovery
        # burst -- a robot stuck in a pileup is stuck either way, and
        # gating this on the recovery branch let a robot cycling in and out
        # of recovery silently delay ever reaching the detour threshold.
        #
        # This isn't only about other robots: a robot can also stall on
        # pure static geometry it's too big for -- e.g. squeezed between a
        # freshly-dropped obstacle and the next pod row at a row-transition
        # diagonal, tight enough for NH-ORCA's static-obstacle lines to
        # nearly zero out its velocity even with no other robot around.
        # D* Lite doesn't know about robot radius at all, so it can route a
        # large robot through a passage a small one clears easily. Rather
        # than model clearance explicitly, just try treating whatever's
        # immediately ahead as temporarily blocked once stuck long enough,
        # regardless of cause, and see if a different route exists.
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

    def _attempt_congestion_detour(self, agent: RobotAgent) -> bool:
        """Treat currently-stuck nearby robots' cells, AND the robot's own
        immediate next path cells, as temporary obstacles and see if
        there's a real alternative route around them. The latter covers
        stalls with no other robot involved at all -- e.g. squeezed by
        static geometry too tight for this robot's own radius, which D*
        Lite can't see coming since it has no notion of robot size. This is
        a one-off scratch plan -- it never touches the shared world or
        other robots' planners, and `agent.planner` (the authoritative D*
        Lite instance against real, static obstacles) is left untouched so
        it resumes normally once the detour commitment window elapses."""
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
                continue  # only route around robots that are themselves stuck
            cell = pos_to_cell(other.position())
            if cell not in (cur_cell, goal):
                phantom_obstacles.add(cell)

        # The next couple of cells this robot was already trying (and
        # failing) to reach -- worth a nudge onto a slightly different cell
        # for genuine multi-robot congestion. (A pure static-geometry squeeze
        # -- too tight for this robot's own radius, which D* Lite can't see
        # coming -- is handled separately via NH_ORCA_STATIC_TIME_HORIZON
        # below, not by blocking cells: blocking a whole neighborhood around
        # a stuck cell risks fully isolating it in an 8-connected grid, and
        # nibbling at one adjacent cell just grazes the same pinch again.)
        upcoming = agent.path[agent.path_index:agent.path_index + 2]
        for cell in upcoming:
            if cell not in (cur_cell, goal):
                phantom_obstacles.add(cell)

        if not phantom_obstacles:
            return False

        scratch = World(self.world.width, self.world.height, strict_diagonal_corners=True)
        scratch.obstacles = set(self.world.obstacles) | phantom_obstacles
        if not scratch.is_free(cur_cell) or not scratch.is_free(goal):
            return False

        detour_path = DStarLite(scratch, cur_cell, goal).get_path()
        if not detour_path or detour_path[-1] != goal:
            return False

        # Reject a "detour" that isn't actually one -- e.g. blocking the one
        # cell ahead can make backtracking the whole way around look like
        # the only option left in the scratch world, even though the real
        # problem (NH-ORCA being overly conservative right at this one
        # spot) will clear on its own shortly. Only commit to a genuinely
        # different route, not a multi-second trip in the wrong direction.
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
        self._reannounce(old_task, exclude=agent)

    def _reannounce(self, task: Task, exclude: RobotAgent) -> None:
        """Release `task` from `exclude` and re-announce it under a fresh id
        (the fleet-wide winning_agents record for the old id still points at
        `exclude`, so reusing it wouldn't let anyone else win it), announced
        by some other agent so `exclude` -- known right now to be a bad
        choice -- doesn't just win its own retry straight back."""
        exclude.cbba.release_task(task.task_id)
        retry = Task(task_id=f"{task.task_id}-retry{self.next_task_id}",
                     pickup=task.pickup, dropoff=task.dropoff,
                     reward=task.reward, created_tick=self.tick)
        self.next_task_id += 1
        candidates = [a for a in self.agents.values() if a is not exclude]
        announcer = random.choice(candidates) if candidates else exclude
        announcer.cbba.announce_task(retry)

    _OBSTACLE_SEARCH_RADIUS = 1.2
    _OBSTACLE_POINT_RADIUS = 0.02
    _MAX_OBSTACLE_POINTS = 6

    def _nearby_static_obstacles(self, pos: tuple[float, float]) -> list[tuple[tuple[float, float], float]]:
        """Nearest point on each occupied cell within range, for NH-ORCA's
        static-obstacle lines. Bounded to a handful of the closest cells so
        the per-frame LP stays small regardless of warehouse size."""
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

        for y in range(GRID_H):
            for x in range(GRID_W):
                c = (x, y)
                rect = cell_rect(c)
                if c in self.pod_cells:
                    pygame.draw.rect(screen, COLOR_POD, rect)
                    pygame.draw.rect(screen, COLOR_POD_EDGE, rect, 1)
                    continue
                if c in self.world.obstacles:
                    pygame.draw.rect(screen, COLOR_OBSTACLE, rect)
                else:
                    pygame.draw.rect(screen, COLOR_FREE, rect)
                pygame.draw.rect(screen, COLOR_GRID_LINE, rect, 1)

        for c in self.pickup_cells:
            pygame.draw.rect(screen, COLOR_PICKUP, cell_rect(c), 2)
            label = self.small_font.render("IN", True, COLOR_PICKUP)
            screen.blit(label, (cell_rect(c).x + 2, cell_rect(c).y + 2))
        for c in self.dropoff_cells:
            pygame.draw.rect(screen, COLOR_DROPOFF, cell_rect(c), 2)
            label = self.small_font.render("OUT", True, COLOR_DROPOFF)
            screen.blit(label, (cell_rect(c).x + 1, cell_rect(c).y + 2))

        for agent in self.agents.values():
            if len(agent.trail) > 1:
                pts = [world_to_px(x, y) for x, y in agent.trail]
                pygame.draw.lines(screen, agent.color, False, pts, 1)
            if agent.path and agent.state in (TaskState.TO_PICKUP, TaskState.TO_DROPOFF):
                for c in agent.path:
                    if c not in self.world.obstacles:
                        r = cell_rect(c).inflate(-CELL_PX * 0.7, -CELL_PX * 0.7)
                        pygame.draw.rect(screen, agent.color, r, 1)

        edge_positions = {rid: a.robot.position() for rid, a in self.agents.items()}
        for yielder, winner in self.resolver.dependency_edges:
            if yielder in edge_positions and winner in edge_positions:
                p1 = world_to_px(*edge_positions[yielder])
                p2 = world_to_px(*edge_positions[winner])
                pygame.draw.line(screen, COLOR_YIELD_EDGE, p1, p2, 2)

        for agent in self.agents.values():
            cx, cy = world_to_px(agent.robot.pose.x, agent.robot.pose.y)
            r_px = int(agent.robot.radius * CELL_PX)
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
                # "No path" means no route to this robot's own pickup/dropoff
                # target -- which may be far away -- not that every adjacent
                # cell is blocked. Draw a dashed line to the actual target so
                # it's clear *what* is unreachable, not just *that* it is.
                if agent.planner is not None:
                    tx, ty = world_to_px(*cell_center(agent.planner.goal))
                    _draw_dashed_line(screen, (cx, cy), (tx, ty), COLOR_WARN)
                    pygame.draw.circle(screen, COLOR_WARN, (tx, ty), 5, 1)
                tag = "NO ROUTE TO TARGET" if agent.unreachable_timer < UNREACHABLE_REASSIGN_S else "REASSIGNING"
                warn_label = self.small_font.render(tag, True, COLOR_WARN)
                screen.blit(warn_label, (cx - warn_label.get_width() // 2, cy + r_px + 4))

            label = self.small_font.render(
                f"{agent.robot_id}:{agent.label} k={self.karma.balance(agent.robot_id)}", True, COLOR_TEXT
            )
            screen.blit(label, (cx - label.get_width() // 2, cy - r_px - 16))

        self._draw_top_bar()
        pygame.display.flip()

    def _draw_top_bar(self) -> None:
        screen = self.screen
        pygame.draw.rect(screen, (10, 11, 14), (0, 0, WINDOW_W, TOP_BAR))

        total_ed_msgs = sum(a.cbba.messages_sent for a in self.agents.values())
        total_periodic_msgs = sum(a.cbba.periodic_message_cost for a in self.agents.values())
        reduction = (1 - total_ed_msgs / total_periodic_msgs) * 100 if total_periodic_msgs else 0.0
        total_expansions = sum(a.planner.node_expansions for a in self.agents.values() if a.planner)

        bundles = " ".join(
            f"{rid}:[{','.join(a.cbba.bundle) or '-'}]" for rid, a in self.agents.items()
        )

        lines = [
            "Click a free cell to toggle an obstacle mid-run.  SPACE: pause/resume   ESC: quit",
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
    FleetSim().run()


if __name__ == "__main__":
    main()
