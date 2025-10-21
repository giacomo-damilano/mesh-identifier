"""Dot detection and pre-analysis utilities."""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import LOGGER
from .config import DetectionConfig


def load_image(path: str) -> np.ndarray:
    """Load a BGR image from *path* and log the result."""
    LOGGER.info("Loading image from %s", path)
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if image.ndim == 3 and image.shape[2] == 4:
        # Preserve fully transparent pixels as white after dropping alpha so
        # they do not appear as black.
        alpha = image[:, :, 3]
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        transparent_mask = alpha == 0
        if np.any(transparent_mask):
            bgr[transparent_mask] = (255, 255, 255)
        image = bgr
    elif image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    LOGGER.info("Image loaded with shape %s", image.shape)
    return image


def _extract_dot_centers(mask: np.ndarray) -> np.ndarray:
    """Return centroids for connected components that match dot heuristics."""

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    centers: List[Tuple[float, float]] = []
    for idx in range(1, num_labels):
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        area = stats[idx, cv2.CC_STAT_AREA]
        if 3 <= w <= 14 and 3 <= h <= 14 and area >= 9:
            centers.append((float(centroids[idx, 0]), float(centroids[idx, 1])))
    return np.array(centers, dtype=np.float32)


def _find_red_dot_candidates(image: np.ndarray) -> np.ndarray:
    """Return centroids of red-looking blobs using HSV thresholding."""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 60, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 60, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return _extract_dot_centers(mask)


def _find_dark_dot_candidates(image: np.ndarray) -> np.ndarray:
    """Return centroids of dark blobs using HSV thresholding mirroring red logic."""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 80, 120])
    mask = cv2.inRange(hsv, lower_dark, upper_dark)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return _extract_dot_centers(mask)


def _merge_candidate_sets(*arrays: np.ndarray, tol: float = 1.5) -> np.ndarray:
    """Merge multiple point arrays while discarding near-duplicate detections."""

    merged: List[Tuple[float, float]] = []
    for arr in arrays:
        for x, y in arr:
            if not merged:
                merged.append((float(x), float(y)))
                continue
            if any((mx - x) ** 2 + (my - y) ** 2 <= tol ** 2 for mx, my in merged):
                continue
            merged.append((float(x), float(y)))
    return np.array(merged, dtype=np.float32)


def detect_dots(image: np.ndarray, config: DetectionConfig) -> np.ndarray:
    """Detect red and dark dots and return their centroids as an ``(N, 2)`` array."""

    LOGGER.info("Detecting dots (red + dark)")
    red_candidates = _find_red_dot_candidates(image)
    LOGGER.info("Red candidate count: %d", len(red_candidates))

    dark_candidates = _find_dark_dot_candidates(image)
    LOGGER.info("Dark candidate count: %d", len(dark_candidates))

    centers_array = _merge_candidate_sets(red_candidates, dark_candidates)
    LOGGER.info("Total unique candidates after merge: %d", len(centers_array))
    return centers_array


def estimate_side_length(centers: np.ndarray, preferred: float) -> Tuple[float, float]:
    """Estimate the typical spacing between dots using nearest neighbours."""
    if len(centers) < 2:
        return preferred, preferred * 0.15
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=min(8, len(centers)))
    flat = dists[:, 1:].reshape(-1)
    flat = flat[np.isfinite(flat) & (flat > 1e-3)]
    if len(flat) == 0:
        return preferred, preferred * 0.15
    median = np.median(flat)
    mad = np.median(np.abs(flat - median)) if len(flat) > 10 else np.std(flat)
    est = float(median)
    spread = float(mad if mad > 1e-3 else median * 0.1)
    LOGGER.info("Estimated side length %.2f (spread %.2f)", est, spread)
    return est, spread


def build_tasks(centers: np.ndarray, max_side: float) -> List[Tuple[int, int, int]]:
    """Return combinations of neighbour indices that seed quadrilateral search."""
    tree = cKDTree(centers)
    neighbor_radius = max_side * np.sqrt(2) * 1.25
    neighbor_lists = [tree.query_ball_point(centers[i], neighbor_radius) for i in range(len(centers))]
    tasks: List[Tuple[int, int, int]] = []
    seen = set()
    for i, neighbors in enumerate(neighbor_lists):
        if len(neighbors) < 2:
            continue
        for j, k in combinations(neighbors, 2):
            triple = tuple(sorted((i, j, k)))
            if triple not in seen:
                seen.add(triple)
                tasks.append(triple)
    LOGGER.info("Generated %d unique triplets", len(tasks))
    return tasks


def summarise_preanalysis(centers: np.ndarray, config: DetectionConfig) -> Dict[str, float]:
    """Compute dynamic thresholds derived from the dot layout."""
    est_side, spread = estimate_side_length(centers, config.preferred_side)
    dynamic_side = max(
        config.preferred_side * 0.8,
        min(est_side * 1.15, config.preferred_side * 1.35),
    )
    angle_tolerance = min(config.max_angle_dev, max(25.0, 2.2 * spread))
    return {
        "estimated_side": float(est_side),
        "spread": float(spread),
        "dynamic_side": float(dynamic_side),
        "angle_tolerance": float(angle_tolerance),
    }


def preanalyse_dots(centers: np.ndarray, config: DetectionConfig) -> Dict[str, np.ndarray]:
    """Return the derived data required for the polygon stage."""
    summary = summarise_preanalysis(centers, config)
    tasks = build_tasks(centers, summary["dynamic_side"])
    payload: Dict[str, np.ndarray] = {
        "centers": centers.astype(np.float32),
        "tasks": np.array(tasks, dtype=np.int32) if tasks else np.empty((0, 3), dtype=np.int32),
        "dynamic_side": np.array(summary["dynamic_side"], dtype=np.float32),
        "spread": np.array(summary["spread"], dtype=np.float32),
        "angle_tolerance": np.array(summary["angle_tolerance"], dtype=np.float32),
        "estimated_side": np.array(summary["estimated_side"], dtype=np.float32),
    }
    return payload


__all__ = [
    "load_image",
    "detect_dots",
    "estimate_side_length",
    "build_tasks",
    "summarise_preanalysis",
    "preanalyse_dots",
]
