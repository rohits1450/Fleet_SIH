"""Build-order step 1: single-robot D* Lite on a static grid.

Click a cell to toggle an obstacle while the robot is en route. Overlay
shows the node-expansion count for the incremental D* Lite replan versus
what a full from-scratch A* search would need on the same map, proving
D* Lite only re-expands the locally-affected region.

Run: python -m scenarios.single_robot_dstar
"""
from __future__ import annotations

import sys

import pygame

from core.planner.dstar_lite import DStarLite, astar_expansions
from core.world import Cell, World

CELL = 22
GRID_W, GRID_H = 38, 26
TOP_BAR = 90
WINDOW_W, WINDOW_H = GRID_W * CELL, GRID_H * CELL + TOP_BAR

MOVE_INTERVAL_MS = 160

COLOR_BG = (18, 20, 24)
COLOR_FREE = (40, 44, 52)
COLOR_GRID_LINE = (30, 33, 39)
COLOR_OBSTACLE = (200, 70, 70)
COLOR_PATH = (70, 130, 200)
COLOR_EXPANDED = (240, 200, 80)
COLOR_START = (90, 200, 120)
COLOR_GOAL = (220, 100, 220)
COLOR_ROBOT = (250, 250, 250)
COLOR_TEXT = (225, 225, 230)
COLOR_WARN = (240, 90, 90)


def cell_rect(c: Cell) -> pygame.Rect:
    x, y = c
    return pygame.Rect(x * CELL, TOP_BAR + y * CELL, CELL, CELL)


def pixel_to_cell(pos: tuple[int, int]) -> Cell | None:
    x, y = pos
    if y < TOP_BAR:
        return None
    return (x // CELL, (y - TOP_BAR) // CELL)


def make_initial_shelves(world: World) -> None:
    # A few warehouse-like shelf rows with aisle gaps, so the grid isn't just
    # open space -- gives D* Lite something to route around from the start.
    for row_y in (5, 6, 12, 13, 19, 20):
        for x in range(3, GRID_W - 3):
            if x % 6 in (4, 5):
                continue  # aisle gap every 6 cells
            world.add_obstacle((x, row_y))


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Phase 1 / Step 1 - D* Lite single robot")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    small_font = pygame.font.SysFont("consolas", 13)

    world = World(GRID_W, GRID_H)
    make_initial_shelves(world)

    start: Cell = (1, 1)
    goal: Cell = (GRID_W - 2, GRID_H - 2)
    world.obstacles.discard(start)
    world.obstacles.discard(goal)

    planner = DStarLite(world, start, goal)
    robot_pos = start
    path = planner.get_path()

    last_dstar_expansions = planner.node_expansions
    last_astar_expansions = None
    last_expanded_cells: list[Cell] = []
    flash_cell: Cell | None = None
    status_message = ""

    move_timer = 0.0
    running = True
    while running:
        dt = clock.tick(60)
        move_timer += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                cell = pixel_to_cell(event.pos)
                if cell is not None and world.in_bounds(cell) and cell not in (robot_pos, goal):
                    changed = world.toggle_obstacle(cell)
                    if changed:
                        flash_cell = cell
                        planner.reset_expansion_counter()
                        planner.notify_obstacles_changed([cell])
                        last_dstar_expansions = planner.node_expansions
                        last_expanded_cells = list(planner.last_expanded)

                        fresh = World(GRID_W, GRID_H)
                        fresh.obstacles = set(world.obstacles)
                        _, last_astar_expansions = astar_expansions(fresh, robot_pos, goal)

                        path = planner.get_path()
                        status_message = "" if path and path[-1] == goal else "NO PATH TO GOAL"

        if move_timer >= MOVE_INTERVAL_MS and status_message == "":
            move_timer = 0.0
            path = planner.get_path()
            if len(path) > 1 and robot_pos != goal:
                robot_pos = path[1]
                planner.update_start(robot_pos)
                planner.compute_shortest_path()
                path = planner.get_path()

        # -- draw --------------------------------------------------------
        screen.fill(COLOR_BG)

        for y in range(GRID_H):
            for x in range(GRID_W):
                c = (x, y)
                rect = cell_rect(c)
                if c in world.obstacles:
                    pygame.draw.rect(screen, COLOR_OBSTACLE, rect)
                else:
                    pygame.draw.rect(screen, COLOR_FREE, rect)
                pygame.draw.rect(screen, COLOR_GRID_LINE, rect, 1)

        for c in last_expanded_cells:
            if c not in world.obstacles:
                pygame.draw.rect(screen, COLOR_EXPANDED, cell_rect(c), 2)

        for c in path:
            if c not in world.obstacles:
                r = cell_rect(c).inflate(-CELL // 2, -CELL // 2)
                pygame.draw.rect(screen, COLOR_PATH, r, border_radius=2)

        pygame.draw.rect(screen, COLOR_START, cell_rect(start), 2)
        pygame.draw.rect(screen, COLOR_GOAL, cell_rect(goal), 3)

        rc = cell_rect(robot_pos)
        pygame.draw.circle(screen, COLOR_ROBOT, rc.center, CELL // 2 - 3)

        if flash_cell is not None:
            pygame.draw.rect(screen, (255, 255, 255), cell_rect(flash_cell), 3)

        pygame.draw.rect(screen, (10, 11, 14), (0, 0, WINDOW_W, TOP_BAR))
        lines = [
            "Click a cell to toggle an obstacle (not on the robot or goal).  Robot auto-navigates via D* Lite.",
            f"Last replan: D* Lite expanded {last_dstar_expansions} node(s)"
            + (f"   |   full A* would expand {last_astar_expansions} node(s)" if last_astar_expansions is not None else "")
            + f"   |   cumulative D* expansions: {planner.node_expansions}",
            f"Robot: {robot_pos}   Goal: {goal}   Path length: {len(path)}",
        ]
        for i, line in enumerate(lines):
            surf = (font if i == 0 else small_font).render(line, True, COLOR_TEXT)
            screen.blit(surf, (10, 6 + i * 22))

        if status_message:
            warn = font.render(status_message, True, COLOR_WARN)
            screen.blit(warn, (WINDOW_W - warn.get_width() - 12, 6))

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
