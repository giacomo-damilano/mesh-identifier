"""Stage 2 alternate: continuous optimisation of quadrilateral constraints."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from detectors import LOGGER
from detectors.config import DetectionConfig
from detectors.io import load_dot_data, save_polygon_payload
from rectangles_autotune import evaluate_configuration, format_results_table, pick_best


PARAM_NAMES: Tuple[str, ...] = (
    "max_angle_dev",
    "side_relaxation",
    "diag_ratio_min",
    "aspect_max",
    "distance_snap_factor",
)

LOWER_BOUNDS = np.array([5.0, 1.1, 1.0, 1.05, 0.05], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dots", default="dot_results.npz", help="Pre-analysed dot dataset")
    parser.add_argument(
        "--output", default="polygon_results_auto_opt.npz", help="Where to store the tuned polygons"
    )
    parser.add_argument(
        "--selection",
        choices=["graph", "ranked"],
        default="ranked",
        help="Conflict resolution strategy: degree-based graph or score-ranked greedy",
    )
    parser.add_argument(
        "--method",
        choices=["nelder-mead", "powell"],
        default="nelder-mead",
        help="Optimisation algorithm to explore the constraint space",
    )
    parser.add_argument("--max-iter", type=int, default=120, help="Maximum optimiser iterations")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Termination tolerance passed to scipy.optimize.minimize",
    )
    parser.add_argument(
        "--initial-step",
        type=float,
        default=0.5,
        help="Initial simplex step (Nelder-Mead) or direction scale (Powell)",
    )
    parser.add_argument("--angle-max", type=float, default=None, help="Upper bound for max_angle_dev")
    parser.add_argument("--side-max", type=float, default=None, help="Upper bound for side_relaxation")
    parser.add_argument("--diag-max", type=float, default=None, help="Upper bound for diag_ratio_min")
    parser.add_argument("--aspect-max", type=float, default=None, help="Upper bound for aspect ratio")
    parser.add_argument(
        "--snap-max",
        type=float,
        default=None,
        help="Upper bound for distance snap factor",
    )
    return parser.parse_args()


def clamp_vector(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.clip(values, lower, upper)


def build_overrides(vector: Sequence[float]) -> Dict[str, float]:
    return {name: float(value) for name, value in zip(PARAM_NAMES, vector)}


def result_key(overrides: Dict[str, float]) -> Tuple[float, ...]:
    return tuple(round(overrides[name], 6) for name in PARAM_NAMES)


def configure_bounds(base: DetectionConfig, args: argparse.Namespace) -> np.ndarray:
    defaults = np.array(
        [
            base.max_angle_dev * 1.6,
            base.side_relaxation + 0.6,
            base.diag_ratio_min + 0.4,
            base.aspect_max + 1.0,
            base.distance_snap_factor + 0.5,
        ],
        dtype=float,
    )
    upper = defaults.copy()
    if args.angle_max:
        upper[0] = args.angle_max
    if args.side_max:
        upper[1] = args.side_max
    if args.diag_max:
        upper[2] = args.diag_max
    if args.aspect_max:
        upper[3] = args.aspect_max
    if args.snap_max:
        upper[4] = args.snap_max
    return np.maximum(upper, LOWER_BOUNDS + 1e-6)


def optimise_constraints(
    centers: np.ndarray,
    tasks: Sequence[Tuple[int, int, int]],
    dynamic_side: float,
    config: DetectionConfig,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    upper_bounds = configure_bounds(config, args)
    lower_bounds = np.minimum(LOWER_BOUNDS, upper_bounds)

    base_vector = np.array(
        [
            config.max_angle_dev,
            config.side_relaxation,
            config.diag_ratio_min,
            config.aspect_max,
            config.distance_snap_factor,
        ],
        dtype=float,
    )

    evaluations: Dict[Tuple[float, ...], Dict[str, object]] = {}
    order: List[Tuple[float, ...]] = []

    baseline = evaluate_configuration(centers, tasks, dynamic_side, config, {}, args.selection)
    key = result_key(baseline["overrides"] or build_overrides(base_vector))
    evaluations[key] = baseline
    order.append(key)
    best = baseline

    def register(result: Dict[str, object]) -> None:
        nonlocal best
        overrides = result["overrides"] or build_overrides(base_vector)
        key = result_key(overrides)
        existing = evaluations.get(key)
        if existing is None:
            evaluations[key] = result
            order.append(key)
        else:
            chosen = pick_best([existing, result])
            evaluations[key] = chosen
        best = pick_best([best, result])

    def objective(vector: np.ndarray) -> float:
        clamped = clamp_vector(vector, lower_bounds, upper_bounds)
        overrides = build_overrides(clamped)
        result = evaluate_configuration(centers, tasks, dynamic_side, config, overrides, args.selection)
        register(result)
        score = (
            -result["polygon_count"] * 1000.0
            + result["mean_score"] * 10.0
            - result["total_area"] / 1e5
        )
        LOGGER.info("Objective score %.3f for overrides %s", score, overrides)
        return score

    method = args.method
    options = {"maxiter": args.max_iter, "tol": args.tol}

    if method == "nelder-mead":
        step = max(args.initial_step, 1e-4)
        simplex = [base_vector]
        for idx in range(len(base_vector)):
            vertex = base_vector.copy()
            vertex[idx] += step
            vertex = clamp_vector(vertex, lower_bounds, upper_bounds)
            simplex.append(vertex)
        options["initial_simplex"] = np.vstack(simplex)

    LOGGER.info(
        "Starting %s optimisation from %s with bounds [%s, %s]",
        method,
        base_vector.tolist(),
        lower_bounds.tolist(),
        upper_bounds.tolist(),
    )

    result = minimize(
        objective,
        base_vector,
        method="Nelder-Mead" if method == "nelder-mead" else "Powell",
        options=options,
    )

    LOGGER.info("Optimisation finished: success=%s, message=%s", result.success, result.message)
    register(best)

    ordered_results = [evaluations[k] for k in order]
    return best, ordered_results


def main() -> None:
    args = parse_args()
    config = DetectionConfig(dot_data_path=args.dots, polygon_data_path=args.output)

    dot_payload = load_dot_data(Path(config.dot_data_path))
    centers = dot_payload["centers"]
    tasks = dot_payload["tasks"]
    dynamic_side = dot_payload["dynamic_side"]
    spread = dot_payload["spread"]
    config.max_angle_dev = dot_payload["angle_tolerance"]

    LOGGER.info(
        "Loaded %d centers with %d triplets (dynamic side %.2f, spread %.2f)",
        len(centers),
        len(tasks),
        dynamic_side,
        spread,
    )

    best, evaluations = optimise_constraints(centers, tasks, dynamic_side, config, args)
    table = format_results_table(evaluations)
    LOGGER.info("Optimisation summary:\n%s", table)

    metadata: Dict[str, object] = {
        "image_path": dot_payload["image_path"],
        "image_height": dot_payload["image_height"],
        "image_width": dot_payload["image_width"],
        "dynamic_side": dynamic_side,
        "angle_tolerance": best["config"].max_angle_dev,
        "estimated_side": dot_payload["estimated_side"],
        "spread": spread,
        "side_relaxation": best["config"].side_relaxation,
        "diag_ratio_min": best["config"].diag_ratio_min,
        "aspect_max": best["config"].aspect_max,
        "distance_snap_factor": best["config"].distance_snap_factor,
        "selection_mode": args.selection,
        "optimizer": args.method,
        "max_iter": args.max_iter,
        "tolerance": args.tol,
        "search_log": table,
    }

    save_polygon_payload(
        Path(config.polygon_data_path),
        centers,
        best["polygons"],
        best["scores"],
        metadata,
    )


if __name__ == "__main__":
    main()
