"""Polygon candidate generation and selection."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

from . import LOGGER
from .config import DetectionConfig
from .geometry import (
    angle_deg,
    convexify,
    order_clockwise,
    polygon_score,
    side_lengths,
    triangle_area,
)


def process_batch(
    batch: Sequence[Tuple[int, int, int]],
    centers: np.ndarray,
    config: DetectionConfig,
    snap_tree: cKDTree,
    dynamic_side: float,
) -> Tuple[List[Tuple], Counter]:
    """Evaluate a batch of triplets and return valid quadrilateral candidates."""

    results: List[Tuple] = []
    diag_counter: Counter = Counter()
    base_side = dynamic_side
    side_cap = base_side * config.side_relaxation
    area_threshold = (base_side ** 2) * config.min_area_ratio
    snap_tol = max(12.0, base_side * config.distance_snap_factor)

    for i, j, k in batch:
        a, b, c = centers[i], centers[j], centers[k]
        tri_area = triangle_area(a, b, c)
        diag_counter["triplets"] += 1
        if tri_area < area_threshold:
            diag_counter["area_skip"] += 1
            continue

        dists = np.array([
            np.linalg.norm(a - b),
            np.linalg.norm(b - c),
            np.linalg.norm(c - a),
        ])
        if np.sum(dists <= side_cap) < 2:
            diag_counter["long_side_skip"] += 1
            continue

        used = False
        for pivot, first, third in ((b, a, c), (a, b, c), (c, a, b)):
            ang = angle_deg(first, pivot, third)
            if abs(ang - config.preferred_angle) > config.max_angle_dev:
                diag_counter["angle_skip"] += 1
                continue

            d_est = first + third - pivot
            dist, idx = snap_tree.query(d_est, k=1, distance_upper_bound=snap_tol)
            if not np.isfinite(dist) or idx >= len(centers) or idx in (i, j, k):
                diag_counter["snap_fail"] += 1
                continue

            quad_idx = tuple(sorted((i, j, k, idx)))
            pts = order_clockwise(centers[list(quad_idx)].copy())
            sides = side_lengths(pts)
            mean_side = float(np.mean(sides))
            if mean_side > side_cap:
                diag_counter["mean_side_skip"] += 1
                continue

            diag1 = np.linalg.norm(pts[0] - pts[2])
            diag2 = np.linalg.norm(pts[1] - pts[3])
            if min(diag1, diag2) < mean_side * config.diag_ratio_min:
                diag_counter["diag_skip"] += 1
                continue

            aspect = max(sides) / max(1e-6, min(sides))
            if aspect > config.aspect_max:
                diag_counter["aspect_skip"] += 1
                continue

            score = polygon_score(pts, config.preferred_angle)
            area = abs(cv2.contourArea(pts.astype(np.float32)))
            results.append((quad_idx, pts, score, area))
            diag_counter["accepted"] += 1
            used = True
            break

        if not used:
            diag_counter["pivot_exhausted"] += 1

    return results, diag_counter


def build_candidates(
    centers: np.ndarray,
    tasks: Sequence[Tuple[int, int, int]],
    config: DetectionConfig,
    dynamic_side: float,
) -> Dict[Tuple[int, int, int, int], Dict]:
    """Return a mapping of unique quadrilateral index tuples to their metrics."""

    if not len(tasks):
        return {}

    batches = [
        tasks[i : i + config.max_triplets_per_batch]
        for i in range(0, len(tasks), config.max_triplets_per_batch)
    ]
    LOGGER.info(
        "Processing %d batches with up to %d triplets each",
        len(batches),
        config.max_triplets_per_batch,
    )

    snap_tree = cKDTree(centers)
    workers = min(config.max_workers, len(batches))
    LOGGER.info("Using %d worker processes", workers)

    parallel = Parallel(n_jobs=workers, backend="loky", verbose=0)
    iterator = (
        delayed(process_batch)(batch, centers, config, snap_tree, dynamic_side)
        for batch in batches
    )

    aggregated: Dict[Tuple[int, int, int, int], Dict] = {}
    diag_total: Counter = Counter()

    for batch_results, diag in parallel(iterator):
        diag_total.update(diag)
        for key, pts, score, area in batch_results:
            if key not in aggregated or score < aggregated[key]["score"]:
                aggregated[key] = {
                    "indices": key,
                    "points": pts,
                    "score": score,
                    "area": area,
                }

    LOGGER.info("Candidate quadrilaterals: %d", len(aggregated))
    LOGGER.info("Diagnostics: %s", dict(diag_total))
    return aggregated


def build_conflict_graph(
    candidates: Sequence[Dict],
    preferred_side: float,
) -> Tuple[List[Dict], List[set]]:
    """Return conflict-graph nodes and adjacency for overlap resolution."""

    cell_size = max(8, int(preferred_side // 4))
    area_eps = 0.4
    LOGGER.info("Building conflict graph with cell size %d", cell_size)

    items: List[Dict] = []
    for idx, candidate in enumerate(candidates):
        poly = convexify(candidate["points"])
        x0, y0 = np.min(poly, axis=0)
        x1, y1 = np.max(poly, axis=0)
        items.append(
            {
                "idx": idx,
                "points": poly,
                "bbox": (x0, y0, x1, y1),
                "score": candidate["score"],
                "area": candidate["area"],
            }
        )

    grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    def iter_cells(bbox: Tuple[float, float, float, float]):
        x0, y0, x1, y1 = bbox
        for cx in range(int(x0 // cell_size), int(x1 // cell_size) + 1):
            for cy in range(int(y0 // cell_size), int(y1 // cell_size) + 1):
                yield (cx, cy)

    for idx, item in enumerate(items):
        for cell in iter_cells(item["bbox"]):
            grid[cell].append(idx)

    neighbors: List[set] = [set() for _ in items]
    checks = 0

    def bboxes_overlap(b1, b2):
        x0a, y0a, x1a, y1a = b1
        x0b, y0b, x1b, y1b = b2
        return not (x1a <= x0b or x1b <= x0a or y1a <= y0b or y1b <= y0a)

    for idxs in grid.values():
        idxs = sorted(idxs)
        for a in range(len(idxs)):
            i = idxs[a]
            bbox_i = items[i]["bbox"]
            poly_i = items[i]["points"]
            for b in range(a + 1, len(idxs)):
                j = idxs[b]
                bbox_j = items[j]["bbox"]
                if not bboxes_overlap(bbox_i, bbox_j):
                    continue
                poly_j = items[j]["points"]
                area, _ = cv2.intersectConvexConvex(poly_i, poly_j)
                checks += 1
                if area > area_eps:
                    neighbors[i].add(j)
                    neighbors[j].add(i)

    LOGGER.info(
        "Conflict graph built with %d nodes and approx %d overlap checks",
        len(items),
        checks,
    )
    return items, neighbors


def select_polygons(items: Sequence[Dict], neighbors: Sequence[set]) -> List[Dict]:
    """Return an approximate independent set of non-overlapping polygons."""

    degrees = np.array([len(n) for n in neighbors], dtype=np.int32)
    order = np.argsort(degrees)
    selected = np.zeros(len(items), dtype=bool)
    banned = np.zeros(len(items), dtype=bool)

    for idx in order:
        if banned[idx]:
            continue
        selected[idx] = True
        for j in neighbors[idx]:
            banned[j] = True

    chosen = [items[i] for i in range(len(items)) if selected[i]]
    LOGGER.info("Selected %d non-overlapping polygons", len(chosen))
    return chosen


__all__ = [
    "process_batch",
    "build_candidates",
    "build_conflict_graph",
    "select_polygons",
]
