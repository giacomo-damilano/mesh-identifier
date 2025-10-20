"""Geometric helper functions for polygon processing."""
from __future__ import annotations

import cv2
import numpy as np


def angle_deg(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Return the angle between the segments ``p1->p2`` and ``p3->p2`` in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def order_clockwise(points: np.ndarray) -> np.ndarray:
    """Return a copy of ``points`` sorted clockwise around their centroid."""
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles)
    return points[order]


def side_lengths(points: np.ndarray) -> np.ndarray:
    """Return the length of each edge in the quadrilateral ``points``."""
    return np.array([
        np.linalg.norm(points[i] - points[(i + 1) % 4])
        for i in range(4)
    ])


def triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the area of the triangle ``abc``."""
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def polygon_score(points: np.ndarray, preferred_angle: float) -> float:
    """Return a heuristic score for how rectangular the quadrilateral appears."""
    sides = side_lengths(points)
    diag1 = np.linalg.norm(points[0] - points[2])
    diag2 = np.linalg.norm(points[1] - points[3])
    angle_penalty = np.mean([
        abs(angle_deg(points[(i - 1) % 4], points[i], points[(i + 1) % 4]) - preferred_angle)
        for i in range(4)
    ])
    diag_penalty = abs(diag1 - diag2)
    opp_penalty = abs(sides[0] - sides[2]) + abs(sides[1] - sides[3])
    aspect_penalty = abs((sides[0] + sides[2]) - (sides[1] + sides[3]))
    return float(angle_penalty + 0.6 * diag_penalty + 0.4 * opp_penalty + 0.25 * aspect_penalty)


def convexify(points: np.ndarray) -> np.ndarray:
    """Return the convex hull of ``points`` as a float32 array."""
    hull = cv2.convexHull(points.astype(np.float32))
    return hull.reshape(-1, 2).astype(np.float32)


__all__ = [
    "angle_deg",
    "order_clockwise",
    "side_lengths",
    "triangle_area",
    "polygon_score",
    "convexify",
]
