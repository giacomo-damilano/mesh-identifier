import logging
import multiprocessing
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from joblib import Parallel, delayed
from pytoshop.user import nested_layers
from scipy.spatial import cKDTree


# =============================================================================
# Logging
# =============================================================================
LOG_FORMAT = "[%(asctime)s][%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOGGER = logging.getLogger("rectangles")


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class DetectionConfig:
    image_path: str = "back_scheme.png"
    output_path: str = "back_scheme_detected.png"
    dot_radius: int = 6
    line_thickness: int = 2
    dot_color_bgr: Tuple[int, int, int] = (0, 0, 255)
    polygon_color_bgr: Tuple[int, int, int] = (0, 255, 0)
    preferred_side: float = 150.0
    preferred_angle: float = 90.0
    max_triplets_per_batch: int = 160
    min_area_ratio: float = 0.015
    max_angle_dev: float = 38.0
    side_relaxation: float = 1.55
    diag_ratio_min: float = 1.05
    aspect_max: float = 1.45
    distance_snap_factor: float = 0.2
    max_workers: int = field(default_factory=lambda: max(1, multiprocessing.cpu_count() - 1))


CONFIG = DetectionConfig()


# =============================================================================
# Geometry helpers
# =============================================================================
def angle_deg(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def order_clockwise(points: np.ndarray) -> np.ndarray:
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles)
    return points[order]


def side_lengths(points: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.norm(points[i] - points[(i + 1) % 4]) for i in range(4)])


def triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def polygon_score(points: np.ndarray, preferred_angle: float) -> float:
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
    return angle_penalty + 0.6 * diag_penalty + 0.4 * opp_penalty + 0.25 * aspect_penalty


# =============================================================================
# Stage 1: load image & detect red dots
# =============================================================================
def load_image(path: str) -> np.ndarray:
    LOGGER.info("Loading image from %s", path)
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    LOGGER.info("Image loaded with shape %s", image.shape)
    return image


def detect_dots(image: np.ndarray, config: DetectionConfig) -> np.ndarray:
    LOGGER.info("Detecting dots using HSV thresholding")
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

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    centers: List[Tuple[float, float]] = []
    for idx in range(1, num_labels):
        w, h, area = stats[idx, cv2.CC_STAT_WIDTH], stats[idx, cv2.CC_STAT_HEIGHT], stats[idx, cv2.CC_STAT_AREA]
        if 4 <= w <= 13 and 4 <= h <= 13 and area >= 15:
            centers.append((float(centroids[idx, 0]), float(centroids[idx, 1])))
    centers_array = np.array(centers, dtype=np.float32)
    LOGGER.info("Detected %d candidate dots", len(centers_array))
    return centers_array


# =============================================================================
# Stage 2: auto-parameter tuning
# =============================================================================
def estimate_side_length(centers: np.ndarray, preferred: float) -> Tuple[float, float]:
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


# =============================================================================
# Stage 3: candidate quadrilaterals
# =============================================================================
def process_batch(batch: Sequence[Tuple[int, int, int]], centers: np.ndarray, config: DetectionConfig,
                  snap_tree: cKDTree, dynamic_side: float, dynamic_spread: float) -> Tuple[List[Tuple], Counter]:
    results: List[Tuple] = []
    diag_counter: Counter = Counter()
    base_side = dynamic_side
    side_cap = base_side * config.side_relaxation
    area_threshold = (base_side ** 2) * config.min_area_ratio
    snap_tol = max(12.0, base_side * config.distance_snap_factor)

    for i, j, k in batch:
        a, b, c = centers[i], centers[j], centers[k]
        tri_area = triangle_area(a, b, c)
        diag_counter['triplets'] += 1
        if tri_area < area_threshold:
            diag_counter['area_skip'] += 1
            continue

        dists = np.array([
            np.linalg.norm(a - b),
            np.linalg.norm(b - c),
            np.linalg.norm(c - a),
        ])
        if np.sum(dists <= side_cap) < 2:
            diag_counter['long_side_skip'] += 1
            continue

        used = False
        for pivot, first, third in ((b, a, c), (a, b, c), (c, a, b)):
            ang = angle_deg(first, pivot, third)
            if abs(ang - config.preferred_angle) > config.max_angle_dev:
                diag_counter['angle_skip'] += 1
                continue

            d_est = first + third - pivot
            dist, idx = snap_tree.query(d_est, k=1, distance_upper_bound=snap_tol)
            if not np.isfinite(dist) or idx >= len(centers) or idx in (i, j, k):
                diag_counter['snap_fail'] += 1
                continue

            quad_idx = tuple(sorted((i, j, k, idx)))
            pts = order_clockwise(centers[list(quad_idx)].copy())
            sides = side_lengths(pts)
            mean_side = float(np.mean(sides))
            if mean_side > side_cap:
                diag_counter['mean_side_skip'] += 1
                continue

            diag1 = np.linalg.norm(pts[0] - pts[2])
            diag2 = np.linalg.norm(pts[1] - pts[3])
            if min(diag1, diag2) < mean_side * config.diag_ratio_min:
                diag_counter['diag_skip'] += 1
                continue

            aspect = max(sides) / max(1e-6, min(sides))
            if aspect > config.aspect_max:
                diag_counter['aspect_skip'] += 1
                continue

            score = polygon_score(pts, config.preferred_angle)
            area = abs(cv2.contourArea(pts.astype(np.float32)))
            results.append((quad_idx, pts, score, area))
            diag_counter['accepted'] += 1
            used = True
            break

        if not used:
            diag_counter['pivot_exhausted'] += 1

    return results, diag_counter


def build_candidates(centers: np.ndarray, tasks: Sequence[Tuple[int, int, int]], config: DetectionConfig,
                     dynamic_side: float, dynamic_spread: float) -> Dict[Tuple[int, int, int, int], Dict]:
    if not tasks:
        return {}

    batches = [tasks[i:i + config.max_triplets_per_batch] for i in range(0, len(tasks), config.max_triplets_per_batch)]
    LOGGER.info("Processing %d batches with up to %d triplets each", len(batches), config.max_triplets_per_batch)

    snap_tree = cKDTree(centers)
    workers = min(config.max_workers, len(batches))
    LOGGER.info("Using %d worker processes", workers)

    parallel = Parallel(n_jobs=workers, backend="loky", verbose=0)
    iterator = (delayed(process_batch)(batch, centers, config, snap_tree, dynamic_side, dynamic_spread)
                for batch in batches)

    aggregated: Dict[Tuple[int, int, int, int], Dict] = {}
    diag_total: Counter = Counter()

    for batch_results, diag in parallel(iterator):
        diag_total.update(diag)
        for key, pts, score, area in batch_results:
            if key not in aggregated or score < aggregated[key]['score']:
                aggregated[key] = {'indices': key, 'points': pts, 'score': score, 'area': area}

    LOGGER.info("Candidate quadrilaterals: %d", len(aggregated))
    LOGGER.info("Diagnostics: %s", dict(diag_total))
    return aggregated


# =============================================================================
# Stage 4: conflict graph & independent set
# =============================================================================
def convexify(points: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(points.astype(np.float32))
    return hull.reshape(-1, 2).astype(np.float32)


def build_conflict_graph(candidates: Sequence[Dict], preferred_side: float) -> Tuple[List[Dict], List[set]]:
    cell_size = max(8, int(preferred_side // 4))
    area_eps = 0.4
    LOGGER.info("Building conflict graph with cell size %d", cell_size)

    items: List[Dict] = []
    for idx, candidate in enumerate(candidates):
        poly = convexify(candidate['points'])
        x0, y0 = np.min(poly, axis=0)
        x1, y1 = np.max(poly, axis=0)
        items.append({
            'idx': idx,
            'points': poly,
            'bbox': (x0, y0, x1, y1),
            'score': candidate['score'],
            'area': candidate['area'],
        })

    grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    def iter_cells(bbox: Tuple[float, float, float, float]):
        x0, y0, x1, y1 = bbox
        for cx in range(int(x0 // cell_size), int(x1 // cell_size) + 1):
            for cy in range(int(y0 // cell_size), int(y1 // cell_size) + 1):
                yield (cx, cy)

    for idx, item in enumerate(items):
        for cell in iter_cells(item['bbox']):
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
            bbox_i = items[i]['bbox']
            poly_i = items[i]['points']
            for b in range(a + 1, len(idxs)):
                j = idxs[b]
                bbox_j = items[j]['bbox']
                if not bboxes_overlap(bbox_i, bbox_j):
                    continue
                poly_j = items[j]['points']
                area, _ = cv2.intersectConvexConvex(poly_i, poly_j)
                checks += 1
                if area > area_eps:
                    neighbors[i].add(j)
                    neighbors[j].add(i)

    LOGGER.info("Conflict graph built with %d nodes and approx %d overlap checks", len(items), checks)
    return items, neighbors


def select_polygons(items: Sequence[Dict], neighbors: Sequence[set]) -> List[Dict]:
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


# =============================================================================
# Stage 5: rendering & export
# =============================================================================
def render_layers(image: np.ndarray, centers: np.ndarray, polygons: Sequence[Dict], config: DetectionConfig) -> None:
    LOGGER.info("Rendering output layers")
    lines = np.zeros_like(image)
    for poly in polygons:
        cv2.polylines(lines, [poly['points'].astype(np.int32)], True, config.polygon_color_bgr, config.line_thickness)
    dots = np.zeros_like(image)
    for x, y in centers:
        cv2.circle(dots, (int(x), int(y)), config.dot_radius, config.dot_color_bgr, -1)

    final = image.copy()
    final = cv2.add(final, lines)
    final = cv2.add(final, dots)

    cv2.imwrite(config.output_path, final)
    LOGGER.info("Saved combined output to %s", config.output_path)

    def cv_to_rgba(arr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGBA).astype(np.uint8)

    background = cv_to_rgba(image)
    lines_layer = cv_to_rgba(lines)
    dots_layer = cv_to_rgba(dots)

    for data, name in [(background, "Background"), (lines_layer, "Lines"), (dots_layer, "Dots")]:
        if data.ndim != 3 or data.shape[2] != 4:
            raise ValueError(f"Layer {name} is not RGBA")
        if data.dtype != np.uint8:
            raise ValueError(f"Layer {name} must be uint8")

    layers = [
        nested_layers.Image(name="Background", image=background),
        nested_layers.Image(name="Lines", image=lines_layer),
        nested_layers.Image(name="Dots", image=dots_layer),
    ]

    psd = nested_layers.nested_layers_to_psd(layers)
    psd_path = config.output_path.replace(".png", "_layers.psd")
    try:
        with open(psd_path, "wb") as handle:
            psd.write(handle)
        if not os.path.exists(psd_path) or os.path.getsize(psd_path) < 1024:
            raise IOError("PSD export produced an empty file")
        LOGGER.info("Saved layered PSD to %s", psd_path)
    except Exception as exc:
        LOGGER.warning("PSD export failed: %s", exc)


# =============================================================================
# Driver
# =============================================================================
def main(config: DetectionConfig = CONFIG) -> None:
    image = load_image(config.image_path)
    centers = detect_dots(image, config)
    if len(centers) < 4:
        raise RuntimeError("Insufficient dots detected")

    est_side, spread = estimate_side_length(centers, config.preferred_side)
    dynamic_side = max(config.preferred_side * 0.8, min(est_side * 1.15, config.preferred_side * 1.35))
    config.max_angle_dev = min(config.max_angle_dev, max(25.0, 2.2 * spread))
    LOGGER.info("Dynamic side %.2f, angle tolerance %.2f", dynamic_side, config.max_angle_dev)

    tasks = build_tasks(centers, dynamic_side)
    candidates_map = build_candidates(centers, tasks, config, dynamic_side, spread)
    items, neighbors = build_conflict_graph(list(candidates_map.values()), dynamic_side)
    selected = select_polygons(items, neighbors)
    LOGGER.info("Final polygon count: %d", len(selected))

    render_layers(image, centers, selected, config)


if __name__ == "__main__":
    main()
