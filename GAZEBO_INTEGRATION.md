# From 2D Pygame to 3D Gazebo: integration guide

This document is for whoever builds the 3D Gazebo/ROS 2 simulation. It
explains what exists in this repo, what ports over unchanged, and exactly
what needs to be built new to plug the same algorithms into a real physics
simulation.

## 1. What this repo is (final state, Phase 1)

A 2D validated reference implementation of a decentralized warehouse fleet:
D* Lite (global planning) + NH-ORCA (local collision avoidance) +
CBBA/ED-CBBA (decentralized task bidding) + Karma-weighted MD-PIBT-style
conflict resolution, coordinated over either an in-process bus or **real
Zenoh sessions** (genuinely separate OS processes, no shared memory, no
central server).

Beyond the original four algorithms, hardening `scenarios/fleet_sim.py`
against real multi-robot traffic (dense pileups, a much larger/slower
"HEAVY" robot alongside standard AMRs, packet loss) surfaced and fixed
several non-obvious failure modes that a Gazebo port needs to know about,
not just the four algorithms in isolation:

- **NH-ORCA needs two different time horizons.** A single shared horizon
  for both robot-robot and static-obstacle avoidance lines badly overstates
  a static corner's threat for a large/slow robot at a tight transition
  (ORCA's velocity obstacle is a *linear* extrapolation of current
  velocity -- fine for reacting early to another robot's true motion,
  wrong for a wall the robot's curved path clears safely). See
  `nh_orca_velocity(..., static_time_horizon=...)`.
- **A stationary pileup needs its own arbitration trigger.** MD-PIBT-style
  conflict detection based on "is either robot still closing" goes silent
  once everyone's already slowed to near-zero (relative velocity ~0 isn't
  negative) -- exactly when a decision is needed most. See
  `ConflictResolver.stuck_radius`.
- **A pairwise yield decision needs hysteresis, and the right amount of it
  is not intuitive.** Re-deciding every tick flickers the winner every
  tick (each decision pays karma, which flips the next comparison). Adding
  a fixed hold fixes that, but with equal deltas the decision providably
  flips back to a tie -- then the same winner -- every time it's
  re-evaluated regardless of hold length, so the hold is picking a resonance
  period against the sim's OTHER timers, not a monotonic "longer is safer"
  dial. The value here (1.0s) was found by sweeping against the full sim
  across many seeds, not derived analytically -- re-sweep it if you port
  the mechanism into a system with different control-loop timing.
- **D* Lite has no notion of robot radius or other robots at all** -- a
  route it finds can be geometrically valid on the grid while genuinely too
  tight for a large robot, or contested by another robot mid-execution.
  `scenarios/fleet_sim.py`'s congestion-detour mechanism (see Section 4) is
  the piece that reacts to this; it is sim-only logic, not part of
  `core/`, and needs a real equivalent in a Gazebo port.

```
core/                  <- pure algorithm library. NO pygame, NO ROS, NO
                           Gazebo imports anywhere in this tree. This is the
                           part that ports to Gazebo essentially unchanged.
  world.py              Grid model: obstacles, cell adjacency, 8-connected
                         neighbors with physically-safe corner-cutting rules.
  robot.py              Pose + unicycle (diff-drive) kinematics, and the
                         epsilon reference-point math NH-ORCA needs.
  planner/dstar_lite.py D* Lite: incremental replanning, not full re-search.
  avoidance/orca.py     ORCA velocity-obstacle math + the 2D LP solver.
  avoidance/nh_orca.py  Non-holonomic adaptation (reference-point shift +
                         Minkowski radius enlargement) + static-obstacle
                         avoidance (walls, not just other robots), with a
                         separate (shorter) time horizon for static lines.
  allocation/cbba.py    CBBA / ED-CBBA task bidding, over any MessageBus.
                         Thread-safe (a reentrant lock around all state
                         mutation -- needed once callbacks can arrive on a
                         Zenoh background thread, not just synchronously
                         in-process), with anti-entropy resync and a
                         mobility (`can_participate`) gate so a boxed-in
                         robot doesn't win a task it can't execute.
  conflict/karma.py     Karma ledger: pairwise yield decisions, fairness.
  conflict/mdpibt.py    Priority/dependency-graph conflict resolution built
                         on Karma; cycle (deadlock) detection and breaking;
                         includes the stuck_radius and decision-hold
                         (hysteresis) tuning described above.
  comms/bus.py          MessageBus interface (Zenoh-style key expressions).
  comms/inprocess.py    Synchronous in-process implementation (2D demo).
  comms/zenoh_bus.py    REAL zenoh session implementation -- this is what
                         a Gazebo/ROS 2 port should use directly.
  metrics.py            Fleet metrics (task completion, collisions).

scenarios/              Pygame-specific 2D demos. NONE of this ports to
                         Gazebo -- it's replaced by Gazebo's own renderer
                         and physics. Read it as a REFERENCE for how the
                         core/ pieces get wired together per tick.
  single_robot_dstar.py  Click-to-obstruct single-robot D* Lite demo.
  two_robot_orca.py      Two-robot head-on NH-ORCA demo.
  fleet_sim.py           The integrated one: N heterogeneous robots, pod-
                         grid warehouse layout, full CBBA + Karma + D* Lite
                         + NH-ORCA running together. THIS is the file whose
                         per-tick control loop (`_step_agent_motion`,
                         `_step_agent_task_fsm`) you're translating into
                         ROS 2 node callbacks.

tests/                  Correctness tests + tests/validation/ (a full sweep
                         against the pass/fail criteria in Algovalidations/,
                         including a real multi-process Zenoh harness in
                         tests/validation/zenoh_worker.py -- useful as a
                         template for a ROS 2/Zenoh bridge node).

scripts/zenoh_netem_manual.sh   Real tc-netem network impairment script
                                (needs sudo) for stress-testing comms.
```

All 41 tests pass (`python -m pytest tests/ -v`). See the validation
conversation history / `tests/validation/*.py` docstrings for what each
scenario proves and any known discrepancies with the checklist wording.

## 2. The one thing that must be preserved: the coordinate contract

Everything in `core/` assumes **1 grid cell = 1.0 world unit**, and a cell
`(x, y)` occupies the continuous square `[x, x+1) x [y, y+1)` with its
center at `(x+0.5, y+0.5)` (see `core/world.py`'s docstring and
`fleet_sim.cell_center()`). D* Lite plans in integer cells; NH-ORCA and the
robot kinematics operate in that same continuous frame directly, with no
conversion layer.

**For Gazebo: 1 grid cell = 1 meter.** Whoever builds the SDF world must
place static geometry (pods/shelves) so that a cell `(x, y)`'s occupied
region in the planning grid corresponds exactly to a 1m x 1m x (height)
footprint at world coordinates `(x, y)` to `(x+1, y+1)` in Gazebo. If this
drifts even slightly, D* Lite will plan through space that's actually
occupied in the 3D world, or refuse cells that are actually free.

The exact warehouse layout to replicate is in `scenarios/fleet_sim.py`:

```python
GRID_W, GRID_H = 30, 18          # 30m x 18m floor
POD_ZONE = (3, 2, 27, 14)        # (x0, y0, x1, y1) -- storage zone bounds
POD_SIZE = 2                     # each pod is a 2x2 cell (2m x 2m) block
POD_PERIOD = 4                   # pods repeat every 4 cells -> 2m aisles
```

`make_pods(world)` in that file is the source of truth for exactly which
cells are occupied -- hand the Gazebo dev the *output* of that function
(a `set[(x,y)]`) so they can generate matching SDF `<collision>` boxes
programmatically rather than hand-placing them and risking drift:

```python
from scenarios.fleet_sim import make_pods
from core.world import World
w = World(30, 18)
pod_cells = make_pods(w)   # -> set of (x, y) integer cells to place a 2x2... 
                           # actually 1x1 boxes at each cell; adjacent cells
                           # in the same pod block naturally form the 2x2 pod
```

Pickup ("IN") stations sit at `x=1` and dropoff ("OUT") stations at
`x=GRID_W-2`, at `y in (2, 6, 10, 14)` -- these should become named Gazebo
model locations (or just marked poses) that a perception/logistics layer
resolves to.

## 3. What ports to Gazebo essentially unchanged

Everything in `core/` is pure Python with no simulation-framework
dependency, by design (see `core/__init__.py`'s docstring — this was
planned from the start). Concretely:

- **`core/world.py` (World, D* Lite's grid)** — reuse as-is. Build the
  `World` once from the known static layout (Section 2). If you later want
  live obstacle detection from Gazebo sensors (lidar/depth camera), that
  perception pipeline just needs to call `world.add_obstacle(cell)` /
  `world.toggle_obstacle(cell)` and `planner.notify_obstacles_changed([cell])`
  — exactly what `fleet_sim._handle_click` does today for a mouse click.

- **`core/planner/dstar_lite.py`** — reuse as-is. No changes needed at all;
  it only ever talks to `World`.

- **`core/avoidance/orca.py` + `nh_orca.py`** — reuse as-is. `nh_orca_velocity`
  takes positions/velocities/radii as plain floats and returns `(v, omega)`;
  it has no idea whether those numbers came from a Pygame simulation or
  Gazebo's ground-truth odometry. Feed it real robot poses instead of
  simulated ones and it works unmodified. Do pass `static_time_horizon`
  explicitly (fleet_sim uses 0.4s against a 2.0s robot-robot horizon) --
  the default falls back to sharing the main horizon, which is what
  originally caused the large/slow robot to stall at tight transitions.

- **`core/allocation/cbba.py`** — reuse as-is. `CBBAAgent` only needs a
  `get_position()` callable and a `MessageBus`; swap the position callback
  to read from your localization topic (e.g. `/robotN/odom`) instead of a
  Pygame robot object. Also wire `can_participate` to a real mobility check
  (e.g. "do I have a valid, non-empty plan right now") -- without it, a
  robot that's temporarily boxed in can still win bids purely on distance
  score and immediately have to reject them.

- **`core/conflict/karma.py` + `mdpibt.py`** — reuse as-is. `ConflictResolver.resolve()`
  takes a `dict[str, RobotView]` (id, position, velocity) each tick and
  returns yield decisions; feed it real robot states from odometry. Keep
  `stuck_radius` and `decision_hold_ticks` as tunables, not hardcoded --
  see the note in Section 1. Real Gazebo control-loop timing (likely
  different from this sim's 60Hz) means `decision_hold_ticks` almost
  certainly needs re-sweeping, not copying verbatim.

- **`core/comms/zenoh_bus.py`** — reuse as-is, and this is the important
  one: it's **already real**, not a simulation stand-in. Each ROS 2 node
  (one per robot, matching the "one process per robot" architecture from
  the original spec) opens its own `ZenohBus()` exactly like
  `tests/validation/zenoh_worker.py` does, and coordination (CBBA bids,
  pose broadcasts for NH-ORCA's neighbor list, Karma resolution) happens
  over the real network, independent of whatever ROS 2 topics you use for
  actuation/sensing. This also means a multi-machine setup (robots as
  separate processes, possibly on separate hosts) works with zero changes.

- **`core/robot.py`'s reference-point math** (`reference_point()`,
  `velocity_at_reference_point()`, `body_velocity_from_reference_velocity()`)
  — reuse as-is. This is pure geometry converting between a holonomic
  `(vx, vy)` (what NH-ORCA solves for) and unicycle `(v, omega)` (what a
  diff-drive robot can execute), independent of who's doing the physics.

## 4. What does NOT port, and what replaces it

- **`DiffDriveRobot.step()`** (the `pose.x += v*cos(theta)*dt` integrator)
  — **do not use this in Gazebo.** Gazebo's diff-drive plugin owns the real
  physics; using both would double-integrate and drift. Instead: publish
  the `(v, omega)` that `nh_orca_velocity()` returns as a
  `geometry_msgs/Twist` to the robot's `/cmd_vel`, and read the robot's
  *actual* pose back from Gazebo's odometry (or a ground-truth pose plugin
  for early testing before a real localization stack is wired up). Keep a
  `Pose` object updated from that feedback for `reference_point()` etc. to
  read — just don't call `.step()` on it.

- **Pygame rendering (`scenarios/*.py`'s `_draw` methods)** — fully
  replaced by Gazebo's own rendering. Nothing here ports; read it only to
  understand what state is meaningful to visualize (should-yield
  highlighting, dependency-graph edges, cargo-carrying indicator, D* Lite
  path overlay) if you want equivalent RViz markers.

- **`scenarios/fleet_sim.py`'s per-tick orchestration
  (`FleetSim._step`, `_step_agent_task_fsm`, `_step_agent_motion`)** —
  the *logic* (task state machine: IDLE → TO_PICKUP → PICKING →
  TO_DROPOFF → DROPPING → IDLE; when to call `planner.update_start`,
  `refresh_plan`, `lookahead_target`) is exactly what a per-robot ROS 2
  node's control-loop timer callback should do. The *mechanism* (a single
  Python process stepping a `pygame` loop) does not — each robot becomes
  its own `rclpy` node with its own timer, not a shared loop iteration.
  Three specific behaviors live only in this orchestration layer, not in
  `core/`, and need a real equivalent, not just a port:

  - **Congestion detour** (`_attempt_congestion_detour`): if a robot makes
    near-zero progress for ~1.2s, it builds a one-off scratch `World` with
    nearby stuck robots' cells *and* its own next couple of path cells
    marked as temporary obstacles, re-plans with a fresh `DStarLite` against
    that scratch world, and adopts the result only if it doesn't cost
    meaningfully more than the route it's replacing (`DETOUR_MAX_COST_RATIO`)
    — otherwise a single blocked cell can make "backtrack the whole way
    around" look like the only option and the robot detours pointlessly
    backward. This exists because D* Lite has no notion of robot radius or
    other robots at all; a route it finds can be grid-valid while genuinely
    too tight for a large robot, or freshly contested by another robot.
    None of the shared `self.world` or other robots' planners are touched —
    it's purely local to the stuck robot, for a few seconds.
  - **Stall recovery**: if a robot's heading has drifted far from its
    direction of travel while stuck (a real failure mode of the epsilon
    reference-point inversion under heavy ORCA constraint — see
    `core/avoidance/nh_orca.py`'s docstring), it does a brief burst of pure
    in-place rotation (zero translation, so it can't cause a new collision)
    toward the intended heading before resuming normal driving.
  - **Unreachable-task reassignment**: if D* Lite reports no path at all
    (`path[-1] != goal`, not just "slow"), the robot waits a grace period
    (`UNREACHABLE_REASSIGN_S`) then abandons the task via CBBA
    re-announcement under a fresh id (see `_reannounce`) so another robot
    picks it up, rather than waiting forever on an undeliverable order.

- **Click-to-obstruct interaction** — replace with either (a) a ROS 2
  service/topic that injects a virtual obstacle cell for testing without
  needing physical props, and/or (b) a real perception pipeline that
  detects unexpected obstacles (a dropped pallet, a person) and calls
  `world.add_obstacle()` / `notify_obstacles_changed()` the same way.

## 5. Step-by-step porting checklist

1. **Package `core/`** as an installable Python package (or a plain
   `PYTHONPATH` addition) inside the ROS 2 workspace — it has no ROS
   dependency, so it doesn't need to be a `colcon` package itself, just
   importable from the nodes that are.
2. **Generate the Gazebo world** from `make_pods()`'s output (Section 2) so
   the 3D static geometry and the planning grid agree exactly. Confirm this
   by running a robot's D* Lite plan and visually checking it against the
   Gazebo world in RViz/Gazebo's GUI before wiring up any motion.
3. **Write one ROS 2 node class, launched once per robot** (matching
   `RobotAgent` + the "one process per robot" architecture). Each instance:
   - Opens its own `ZenohBus()`.
   - Constructs a `CBBAAgent` (get_position → its own odometry subscriber).
   - Subscribes to `fleet/*/pose` (see `zenoh_worker.py`) to build the
     `neighbors` list for NH-ORCA and the `RobotView` dict for
     `ConflictResolver`.
   - On a timer (e.g. 20-50 Hz): refresh its D* Lite plan if it moved to a
     new cell, compute the lookahead target, compute preferred velocity,
     run `resolve()` + `nh_orca_velocity()`, publish the resulting Twist.
   - Runs the task FSM exactly like `_step_agent_task_fsm`, but driven by
     real arrival detection (distance to target below threshold) instead
     of a Pygame tick.
4. **Bootstrap with ground-truth poses first** (Gazebo's `p3d`/pose-array
   plugin or a `/gazebo/model_states` bridge), before wiring in a real
   localization stack (AMCL, SLAM) — this isolates "does the algorithm
   work" from "does my localization work," matching how this repo isolated
   D* Lite/NH-ORCA/CBBA/Karma from each other during validation.
5. **Validate against `tests/validation/`** — those tests encode the exact
   pass criteria from `Algovalidations/` at the algorithm level; use them
   as the specification for what "correct" looks like when the same logic
   runs against real Gazebo feedback instead of simulated state. Discrepancies
   are far easier to debug at that level than by only watching the 3D view.
6. **Re-sweep `decision_hold_ticks` (and `stuck_radius` if you change robot
   footprints) against your own control-loop timing** before trusting fleet
   behavior with more than one or two robots. This sim's value (1.0s) was
   found empirically by sweeping against 40 full simulated runs, not derived
   from a formula, and the relationship to hold length is *not* monotonic --
   it resonates with whatever other periodic timers you have (stall-recovery
   burst duration, congestion-detour timeout), so a naive port of the
   number without re-sweeping can silently reintroduce the exact stall/
   flicker/collision failure modes this repo went through several rounds to
   fix. Sweep by tracking, per candidate value, worst-case continuous
   near-zero-velocity streak per robot AND collision count across many
   randomized runs -- optimizing for either alone is what caused the
   regressions here.
7. **Port the three sim-only behaviors from Section 4** (congestion detour,
   stall recovery, unreachable-task reassignment) into the per-robot node --
   they're what actually kept the large/slow robot moving once the four
   core algorithms alone stopped being enough under real multi-robot
   traffic. Validate each the same way this repo did: reproduce a stuck
   robot deliberately (force two robots into a tight encounter, or box one
   in with obstacles), confirm it resolves within a bounded time, and sweep
   across enough randomized scenarios that you're not just confirming the
   one case you built the fix against.
8. **Only then** layer in the stress scenarios (dropped obstacles, a robot
   going offline/on the real network) — `scripts/zenoh_netem_manual.sh` and
   the partition/reconnection pattern in `zenoh_worker.py` transfer directly
   since the comms layer is already real.
