"""Build-order step 2: differential-drive kinematics + NH-ORCA for two
robots on a head-on course.

Two robots swap positions across an open floor. NH-ORCA computes a
collision-free (vx, vy) at each robot's epsilon-offset reference point and
we invert that back to unicycle (v, omega). Overlay tracks collision count
(should stay zero) and path smoothness (running |omega|).

Run: python -m scenarios.two_robot_orca
"""
from __future__ import annotations

import math
import sys
from collections import deque

import pygame

from core.avoidance.nh_orca import DEFAULT_EPSILON, nh_orca_velocity
from core.robot import DiffDriveRobot, Pose

PPM = 70.0  # pixels per meter
WORLD_W_M, WORLD_H_M = 11.0, 6.5
TOP_BAR = 90
WINDOW_W = int(WORLD_W_M * PPM)
WINDOW_H = int(WORLD_H_M * PPM) + TOP_BAR

DT = 1.0 / 60.0
TIME_HORIZON = 2.5
GOAL_TOLERANCE = 0.25

COLOR_BG = (18, 20, 24)
COLOR_FLOOR = (32, 35, 41)
COLOR_TEXT = (225, 225, 230)
COLOR_GOAL_A = (90, 200, 120)
COLOR_GOAL_B = (220, 140, 90)
COLOR_ROBOT_A = (90, 170, 240)
COLOR_ROBOT_B = (240, 130, 130)
COLOR_WARN = (240, 90, 90)


def world_to_px(x: float, y: float) -> tuple[int, int]:
    return int(x * PPM), int(TOP_BAR + y * PPM)


def draw_robot(screen, robot: DiffDriveRobot, color, trail) -> None:
    if len(trail) > 1:
        pts = [world_to_px(x, y) for x, y in trail]
        pygame.draw.lines(screen, color, False, pts, 2)

    cx, cy = world_to_px(robot.pose.x, robot.pose.y)
    r_px = int(robot.radius * PPM)
    pygame.draw.circle(screen, color, (cx, cy), r_px, 2)
    hx = cx + int(math.cos(robot.pose.theta) * r_px)
    hy = cy + int(math.sin(robot.pose.theta) * r_px)
    pygame.draw.line(screen, color, (cx, cy), (hx, hy), 2)


def toward(pos: tuple[float, float], goal: tuple[float, float], speed: float) -> tuple[float, float]:
    dx, dy = goal[0] - pos[0], goal[1] - pos[1]
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return (0.0, 0.0)
    s = min(speed, d * 2.0)  # ease into the goal instead of overshooting
    return (dx / d * s, dy / d * s)


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Phase 1 / Step 2 - NH-ORCA head-on")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    small_font = pygame.font.SysFont("consolas", 13)

    radius = 0.35
    # Exact mirror symmetry is ORCA's known singular case (velocity decays
    # to zero without either agent breaking sideways); a small y offset is
    # what any real course would have anyway.
    robot_a = DiffDriveRobot("A", Pose(1.0, WORLD_H_M / 2 - 0.08, 0.0), radius=radius, max_speed=1.1, max_omega=3.0)
    robot_b = DiffDriveRobot("B", Pose(WORLD_W_M - 1.0, WORLD_H_M / 2, math.pi), radius=radius, max_speed=1.1, max_omega=3.0)
    goal_a = (WORLD_W_M - 1.0, WORLD_H_M / 2)
    goal_b = (1.0, WORLD_H_M / 2)

    trail_a: deque = deque(maxlen=400)
    trail_b: deque = deque(maxlen=400)

    collision_count = 0
    was_colliding = False
    min_dist_seen = float("inf")
    omega_accum = 0.0
    omega_samples = 0

    running = True
    paused = False
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    robot_a.pose = Pose(1.0, WORLD_H_M / 2 - 0.08, 0.0)
                    robot_b.pose = Pose(WORLD_W_M - 1.0, WORLD_H_M / 2, math.pi)
                    robot_a.velocity = robot_b.velocity = (0.0, 0.0)
                    trail_a.clear()
                    trail_b.clear()
                    collision_count = 0
                    min_dist_seen = float("inf")
                    omega_accum = 0.0
                    omega_samples = 0

        if not paused:
            ref_a = robot_a.reference_point(DEFAULT_EPSILON)
            ref_b = robot_b.reference_point(DEFAULT_EPSILON)
            pref_a = toward(ref_a, goal_a, robot_a.max_speed)
            pref_b = toward(ref_b, goal_b, robot_b.max_speed)

            v_a, w_a = nh_orca_velocity(robot_a, [robot_b], pref_a, TIME_HORIZON, DT)
            v_b, w_b = nh_orca_velocity(robot_b, [robot_a], pref_b, TIME_HORIZON, DT)
            robot_a.set_body_velocity(v_a, w_a)
            robot_b.set_body_velocity(v_b, w_b)
            robot_a.step(DT)
            robot_b.step(DT)

            trail_a.append(robot_a.position())
            trail_b.append(robot_b.position())

            d = math.hypot(robot_a.pose.x - robot_b.pose.x, robot_a.pose.y - robot_b.pose.y)
            min_dist_seen = min(min_dist_seen, d)
            colliding = d < (robot_a.radius + robot_b.radius)
            if colliding and not was_colliding:
                collision_count += 1
            was_colliding = colliding

            omega_accum += abs(w_a) + abs(w_b)
            omega_samples += 2

        screen.fill(COLOR_BG)
        pygame.draw.rect(screen, COLOR_FLOOR, (0, TOP_BAR, WINDOW_W, WINDOW_H - TOP_BAR))

        pygame.draw.circle(screen, COLOR_GOAL_A, world_to_px(*goal_a), 8, 2)
        pygame.draw.circle(screen, COLOR_GOAL_B, world_to_px(*goal_b), 8, 2)
        draw_robot(screen, robot_a, COLOR_ROBOT_A, trail_a)
        draw_robot(screen, robot_b, COLOR_ROBOT_B, trail_b)

        pygame.draw.rect(screen, (10, 11, 14), (0, 0, WINDOW_W, TOP_BAR))
        avg_omega = omega_accum / omega_samples if omega_samples else 0.0
        lines = [
            "SPACE: pause/resume   R: reset   ESC: quit",
            f"Min separation so far: {min_dist_seen:.3f} m   (collision threshold: {robot_a.radius + robot_b.radius:.2f} m)",
            f"Collision count: {collision_count}   |   avg |omega| (path smoothness): {avg_omega:.3f} rad/s",
        ]
        for i, line in enumerate(lines):
            color = COLOR_WARN if (i == 2 and collision_count > 0) else COLOR_TEXT
            surf = (font if i == 0 else small_font).render(line, True, color)
            screen.blit(surf, (10, 6 + i * 22))

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
