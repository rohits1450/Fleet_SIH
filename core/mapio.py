"""Load a ROS map_server-style occupancy map (YAML + image) into a World,
plus generic geometry helpers for pulling task/spawn points out of it.

No pygame/ROS/Gazebo imports here, consistent with the rest of core/ --
loading a real map into a World is exactly what a Gazebo-side planner needs
too, not just this repo's pygame demo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from core.world import Cell, World

Side = str  # 'N', 'S', 'E', or 'W'


@dataclass
class MapMeta:
    resolution: float                      # meters/pixel in the source image
    origin: tuple[float, float, float]     # world pose of the image's bottom-left pixel (ROS convention)
    cell_size: float                       # meters/cell in the returned World
    pixels_per_cell: int
    image_size: tuple[int, int]            # (width, height) in pixels


def load_world_from_map(
    yaml_path: str | Path,
    cell_size: float | None = None,
    occupied_frac: float = 0.5,
    min_component_cells: int = 8,
) -> tuple[World, MapMeta]:
    """Build a World from a ROS map_server YAML + its image.

    `cell_size` (meters) downsamples the native pixel grid into coarser
    planning cells via majority vote over each block -- native map
    resolution (commonly ~0.05m/px) makes for a planning grid far bigger
    than a per-tick D* Lite replan needs. Majority vote (rather than "any
    pixel occupied") also naturally filters sub-cell-sized markers (e.g. a
    few pedestrian-spawn dots) that don't cover a meaningful share of a
    cell. `cell_size=None` keeps native pixel resolution.

    `min_component_cells` drops any occupied connected component smaller
    than this from the map entirely (treated as free) -- majority voting
    alone doesn't catch a marker that happens to survive as its own tiny
    cell cluster.
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        meta_raw = yaml.safe_load(f)

    resolution = float(meta_raw["resolution"])
    origin = tuple(meta_raw.get("origin", (0.0, 0.0, 0.0)))
    negate = bool(meta_raw.get("negate", 0))

    image_path = yaml_path.parent / meta_raw["image"]
    if not image_path.exists():
        # The YAML's declared image (typically a .pgm) is sometimes missing
        # in favor of a renamed/exported schematic sitting alongside it --
        # fall back to the one image file in the same directory rather than
        # failing outright.
        candidates = [
            p for p in yaml_path.parent.iterdir()
            if p.suffix.lower() in (".png", ".pgm", ".jpg", ".jpeg")
        ]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Map image {image_path} not found and no unique fallback "
                f"image exists in {yaml_path.parent}"
            )
        image_path = candidates[0]

    lum = _load_luminance(image_path)
    height_px, width_px = lum.shape

    # The YAML's occupied_thresh/free_thresh assume a genuine grayscale
    # occupancy pgm; a schematic rendering standing in for one (as here --
    # see load_world_from_map's docstring on the image fallback) can use a
    # mid-gray shelf fill that those thresholds would call confidently
    # "free" (e.g. a fill at 80% brightness scores 0.2 on the standard
    # occ = (255-p)/255 scale, under BOTH thresholds, when the real .pgm
    # this map was calibrated for presumably drew shelves near-black
    # instead). Thresholds derived from the two YAML values can't fix that
    # -- the mismatch is in the pixel value itself, not the cutoff -- so
    # classify relative to the image's own background shade instead:
    # whichever extreme (brightest, or darkest if negate) represents free
    # space, and treat any pixel that isn't close to it as occupied. This
    # still respects `negate`, and matches plain ROS semantics whenever the
    # image *is* a proper two-level occupancy map.
    bg = float(lum.min()) if negate else float(lum.max())
    free_mask = np.isclose(lum, bg, atol=0.05)
    occupied_mask = ~free_mask

    pixels_per_cell = 1 if cell_size is None else max(1, round(cell_size / resolution))
    grid_w = math.ceil(width_px / pixels_per_cell)
    grid_h = math.ceil(height_px / pixels_per_cell)

    world = World(grid_w, grid_h, strict_diagonal_corners=True)
    for gy in range(grid_h):
        py0, py1 = gy * pixels_per_cell, min((gy + 1) * pixels_per_cell, height_px)
        for gx in range(grid_w):
            px0, px1 = gx * pixels_per_cell, min((gx + 1) * pixels_per_cell, width_px)
            block = occupied_mask[py0:py1, px0:px1]
            if block.size and block.mean() >= occupied_frac:
                world.add_obstacle((gx, gy))

    for comp in connected_components(world):
        if len(comp) < min_component_cells:
            for c in comp:
                world.remove_obstacle(c)

    _fill_corner_notches(world)

    meta = MapMeta(
        resolution=resolution,
        origin=origin,  # type: ignore[arg-type]
        cell_size=cell_size if cell_size is not None else resolution,
        pixels_per_cell=pixels_per_cell,
        image_size=(width_px, height_px),
    )
    return world, meta


def _load_luminance(path: Path) -> np.ndarray:
    import matplotlib.image as mpimg

    arr = mpimg.imread(str(path))
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float64) / 255.0
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    return arr


# -- geometry helpers ---------------------------------------------------

def connected_components(world: World) -> list[set[Cell]]:
    """4-connected components of world.obstacles."""
    visited: set[Cell] = set()
    comps: list[set[Cell]] = []
    for start in world.obstacles:
        if start in visited:
            continue
        comp: set[Cell] = set()
        stack = [start]
        visited.add(start)
        while stack:
            x, y = stack.pop()
            comp.add((x, y))
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in world.obstacles and n not in visited:
                    visited.add(n)
                    stack.append(n)
        comps.append(comp)
    return comps


def _fill_corner_notches(world: World) -> None:
    """Fill free cells that sit in a concave inside corner -- both an
    orthogonal horizontal AND vertical neighbor occupied (or off-grid).

    Downsampling a pixel obstacle rectangle by majority vote can leave its
    corner cell just under the occupied threshold (a corner pixel block is
    the one most likely to be partially covered), producing a single-cell
    notch that world.neighbors()'s strict_diagonal_corners rule already
    refuses to *cut across diagonally* -- but does nothing to stop a robot
    from being routed straight *into* via two orthogonal steps, where NH-
    ORCA then sees close obstacle points on two adjacent sides at once and
    can stall indefinitely (the corner is real geometry, not open floor,
    for any robot with actual radius). This makes the grid's connectivity
    agree with what strict_diagonal_corners already decided was blocked."""
    w, h = world.width, world.height

    def blocked(c: Cell) -> bool:
        # Off-grid doesn't count -- a cell near the map's edge isn't an
        # inside corner of real geometry just because part of its
        # neighborhood falls outside the image.
        return world.in_bounds(c) and c in world.obstacles

    to_fill = []
    for y in range(h):
        for x in range(w):
            c = (x, y)
            if c in world.obstacles:
                continue
            corners = (
                ((x + 1, y), (x, y - 1)), ((x + 1, y), (x, y + 1)),
                ((x - 1, y), (x, y - 1)), ((x - 1, y), (x, y + 1)),
            )
            if any(blocked(a) and blocked(b) for a, b in corners):
                to_fill.append(c)

    for c in to_fill:
        world.add_obstacle(c)


def _bbox(cells: set[Cell]) -> tuple[int, int, int, int]:
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return min(xs), min(ys), max(xs), max(ys)


def classify_wall_and_shelf_components(
    world: World, wall_span_frac: float = 0.6
) -> tuple[list[set[Cell]], list[set[Cell]]]:
    """Split obstacle components into building walls vs. interior shelving.

    A perimeter wall (even broken up by door gaps into several components)
    spans most of the map in at least one dimension; every real shelf is
    much smaller than that. `wall_span_frac` is the fraction of the grid's
    width/height a component's bounding box must reach to count as wall."""
    walls, shelves = [], []
    for comp in connected_components(world):
        x0, y0, x1, y1 = _bbox(comp)
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if w >= wall_span_frac * world.width or h >= wall_span_frac * world.height:
            walls.append(comp)
        else:
            shelves.append(comp)
    return walls, shelves


def boundary_gaps(
    world: World, wall_cells: set[Cell], band: int = 1
) -> list[tuple[Side, int, int, int]]:
    """Find door-sized gaps in the wall's bounding rectangle.

    Returns (side, line_coord, span_start, span_end) tuples -- `line_coord`
    is the wall's row (N/S) or column (E/W) index, and the span is the
    range of the perpendicular axis that is free rather than wall. `band`
    widens the wall-presence check by this many cells to either side of the
    boundary line, tolerating a line that isn't perfectly straight."""
    x0, y0, x1, y1 = _bbox(wall_cells)
    gaps: list[tuple[Side, int, int, int]] = []

    def scan(side: Side, line: int, span: range, cell_at) -> None:
        present = [
            any(cell_at(i, line + d) in wall_cells for d in range(-band, band + 1))
            for i in span
        ]
        n = len(present)
        i = 0
        while i < n:
            if present[i]:
                i += 1
                continue
            j = i
            while j < n and not present[j]:
                j += 1
            if i > 0 and j < n and world.is_free(cell_at(span[i], line)):
                gaps.append((side, line, span[i], span[j - 1]))
            i = j

    scan("N", y0, range(x0, x1 + 1), lambda i, l: (i, l))
    scan("S", y1, range(x0, x1 + 1), lambda i, l: (i, l))
    scan("W", x0, range(y0, y1 + 1), lambda i, l: (l, i))
    scan("E", x1, range(y0, y1 + 1), lambda i, l: (l, i))
    return gaps


def inward_direction(side: Side) -> Cell:
    return {"N": (0, 1), "S": (0, -1), "W": (1, 0), "E": (-1, 0)}[side]


def line_cell(side: Side, line_coord: int, perp_coord: int) -> Cell:
    return (perp_coord, line_coord) if side in ("N", "S") else (line_coord, perp_coord)


def shelf_adjacent_cells(world: World, shelf_cells: set[Cell]) -> dict[Side, list[Cell]]:
    """Free cells orthogonally adjacent to a shelf component, grouped by
    which side of the shelf they sit on -- the aisle cells a robot would
    approach that side from."""
    by_side: dict[Side, list[Cell]] = {}
    for x, y in shelf_cells:
        for side, (nx, ny) in (("N", (x, y - 1)), ("S", (x, y + 1)), ("W", (x - 1, y)), ("E", (x + 1, y))):
            n = (nx, ny)
            if world.is_free(n):
                by_side.setdefault(side, []).append(n)
    return by_side


def find_clear_cells(world: World, radius: int = 1) -> list[Cell]:
    """Free cells whose neighborhood out to `radius` cells (Chebyshev) is
    entirely free -- candidates for robot spawn points, clear of any
    obstacle by more than a single cell."""
    clear = []
    for y in range(world.height):
        for x in range(world.width):
            c = (x, y)
            if not world.is_free(c):
                continue
            if all(
                world.is_free((x + dx, y + dy))
                for dx in range(-radius, radius + 1)
                for dy in range(-radius, radius + 1)
            ):
                clear.append(c)
    return clear
