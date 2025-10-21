"""Stage 1 alternate: tune dot detection thresholds via local optimisation."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

import cv2

from detectors import LOGGER
from detectors.config import DetectionConfig
from detectors.dots import detect_dots, load_image, preanalyse_dots
from dots_detection import save_dot_payload


TARGET_DOT_COUNT = 3240
CLUSTER_DISTANCE = 135.0


@dataclass
class DetectionParameters:
    saturation_min: float
    value_min: float
    area_min: float
    area_max: float

    def sanitise(self) -> "DetectionParameters":
        sat = float(np.clip(self.saturation_min, 10.0, 180.0))
        val = float(np.clip(self.value_min, 10.0, 220.0))
        amin = float(np.clip(self.area_min, 5.0, 150.0))
        amax_floor = amin + 5.0
        amax = float(np.clip(self.area_max, amax_floor, 220.0))
        return DetectionParameters(sat, val, amin, amax)

    def as_vector(self) -> np.ndarray:
        return np.array([self.saturation_min, self.value_min, self.area_min, self.area_max], dtype=np.float64)


def build_mask(image: np.ndarray, params: DetectionParameters) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0.0, params.saturation_min, params.value_min], dtype=np.float32)
    upper_red1 = np.array([10.0, 255.0, 255.0], dtype=np.float32)
    lower_red2 = np.array([170.0, params.saturation_min, params.value_min], dtype=np.float32)
    upper_red2 = np.array([180.0, 255.0, 255.0], dtype=np.float32)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def detect_with_parameters(
    image: np.ndarray,
    params: DetectionParameters,
    scale_factor: float = 1.0,
) -> np.ndarray:
    params = params.sanitise()
    mask = build_mask(image, params)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    centers: List[Tuple[float, float]] = []
    scale_factor = max(scale_factor, 1e-3)
    width_min = max(2, int(round(4 * scale_factor)))
    width_max = max(width_min, int(round(13 * scale_factor)))
    area_scale = scale_factor * scale_factor
    area_min = max(1.0, params.area_min * area_scale)
    area_max = max(area_min + 1.0, params.area_max * area_scale)
    for idx in range(1, num_labels):
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        area = stats[idx, cv2.CC_STAT_AREA]
        if width_min <= w <= width_max and width_min <= h <= width_max and area_min <= area <= area_max:
            centers.append((float(centroids[idx, 0]), float(centroids[idx, 1])))
    return np.array(centers, dtype=np.float32)


def count_close_pairs(points: np.ndarray, distance: float) -> int:
    if len(points) < 2:
        return 0
    tree = cKDTree(points)
    pairs = tree.query_pairs(distance)
    return len(pairs)


def objective(
    raw_params: Iterable[float],
    image: np.ndarray,
    target_count: float,
    distance_threshold: float,
    reference_penalty: float,
    scale_factor: float,
    baseline_pairs: float,
) -> float:
    params = DetectionParameters(*raw_params).sanitise()
    centers = detect_with_parameters(image, params, scale_factor=scale_factor)
    count = len(centers)

    if count < 4:
        return reference_penalty * 10.0

    count_error = (count - target_count) / target_count
    count_penalty = float(count_error * count_error)

    close_pairs = count_close_pairs(centers, distance_threshold * scale_factor)
    excess_pairs = max(0.0, close_pairs - baseline_pairs)
    cluster_penalty = excess_pairs / max(1.0, count)

    total = count_penalty + cluster_penalty

    LOGGER.debug(
        "Objective eval params=%s count=%d close_pairs=%d score=%.6f",
        params,
        count,
        close_pairs,
        total,
    )
    return total


def optimise_parameters(
    image: np.ndarray,
    initial: DetectionParameters,
    method: str,
    target_count: float,
    distance_threshold: float,
    max_iter: int,
    scale_factor: float,
) -> DetectionParameters:
    initial = initial.sanitise()
    baseline_centers = detect_with_parameters(image, initial, scale_factor=scale_factor)
    baseline_pairs = (
        count_close_pairs(baseline_centers, distance_threshold * scale_factor)
        if len(baseline_centers) >= 2
        else 0.0
    )
    reference_penalty = objective(
        initial.as_vector(),
        image,
        target_count,
        distance_threshold,
        reference_penalty=1.0,
        scale_factor=scale_factor,
        baseline_pairs=baseline_pairs,
    )
    LOGGER.info(
        "Starting optimisation with %s method from %s (score %.6f)",
        method,
        initial,
        reference_penalty,
    )

    options = {"maxiter": max_iter} if max_iter > 0 else None

    result = minimize(
        objective,
        initial.as_vector(),
        args=(
            image,
            target_count,
            distance_threshold,
            reference_penalty or 1.0,
            scale_factor,
            baseline_pairs,
        ),
        method=method,
        options=options or {},
    )

    if not result.success:
        LOGGER.warning("Optimiser reported failure: %s", result.message)

    best = DetectionParameters(*result.x).sanitise()
    LOGGER.info("Optimisation complete: %s (score %.6f)", best, result.fun)
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="back_scheme.png", help="Input image path")
    parser.add_argument("--output", default="dot_results_opt.npz", help="Output dataset path")
    parser.add_argument(
        "--method",
        choices=["Nelder-Mead", "Powell"],
        default="Nelder-Mead",
        help="Optimisation strategy",
    )
    parser.add_argument("--target-count", type=float, default=TARGET_DOT_COUNT, help="Expected dot count")
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=CLUSTER_DISTANCE,
        help="Minimum allowed spacing between dots before considering them clustered",
    )
    parser.add_argument("--max-iter", type=int, default=120, help="Limit iterations for the optimiser")
    parser.add_argument(
        "--optim-scale",
        type=float,
        default=0.7,
        help="Resize factor for optimisation evaluations (1.0 = full resolution)",
    )
    parser.add_argument("--initial-s", type=float, default=60.0, help="Initial HSV saturation lower bound")
    parser.add_argument("--initial-v", type=float, default=70.0, help="Initial HSV value lower bound")
    parser.add_argument("--initial-area-min", type=float, default=15.0, help="Initial minimum component area")
    parser.add_argument("--initial-area-max", type=float, default=70.0, help="Initial maximum component area")
    parser.add_argument(
        "--allow-close",
        action="store_true",
        help="Do not raise an error if close dot clusters are detected",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DetectionConfig(image_path=args.image, dot_data_path=args.output)

    image = load_image(config.image_path)
    optim_scale = float(np.clip(args.optim_scale, 0.1, 1.0))
    if optim_scale != 1.0:
        LOGGER.info("Resizing image by factor %.3f for optimisation", optim_scale)
        optim_image = cv2.resize(
            image,
            None,
            fx=optim_scale,
            fy=optim_scale,
            interpolation=cv2.INTER_AREA if optim_scale < 1.0 else cv2.INTER_LINEAR,
        )
    else:
        optim_image = image

    baseline_centers = detect_dots(image, config)
    baseline_pairs = count_close_pairs(baseline_centers, args.distance_threshold)
    LOGGER.info(
        "Baseline detection: %d dots with %d clustered pairs (< %.1fpx)",
        len(baseline_centers),
        baseline_pairs,
        args.distance_threshold,
    )

    initial = DetectionParameters(
        args.initial_s,
        args.initial_v,
        args.initial_area_min,
        args.initial_area_max,
    )

    best = optimise_parameters(
        optim_image,
        initial,
        method=args.method,
        target_count=args.target_count,
        distance_threshold=args.distance_threshold,
        max_iter=args.max_iter,
        scale_factor=optim_scale,
    )

    centers = detect_with_parameters(image, best, scale_factor=1.0)
    if len(centers) < 4:
        raise RuntimeError("Optimised parameters failed to detect sufficient dots")

    close_pairs = count_close_pairs(centers, args.distance_threshold)
    if close_pairs > baseline_pairs and not args.allow_close:
        raise RuntimeError(
            f"Optimised detection produced {close_pairs} clustered dot pairs (< {args.distance_threshold}px), exceeding baseline {baseline_pairs}"
        )

    payload = preanalyse_dots(centers, config)
    payload["optimised_params"] = np.array(best.as_vector(), dtype=np.float32)
    payload["optimiser"] = np.array(args.method)
    payload["baseline_pairs"] = np.array(float(baseline_pairs), dtype=np.float32)
    payload["optim_close_pairs"] = np.array(float(close_pairs), dtype=np.float32)

    save_dot_payload(Path(config.dot_data_path), config.image_path, image.shape[:2], payload)

    LOGGER.info(
        "Optimised detection stored %d dots with %d clustered pairs", len(centers), close_pairs
    )


if __name__ == "__main__":
    main()
