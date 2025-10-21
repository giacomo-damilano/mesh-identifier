"""Stage 2 alternate: iterate constraint sets to maximise polygon coverage."""
from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from detectors import LOGGER
from detectors.config import DetectionConfig
from detectors.io import load_dot_data, save_polygon_payload
from detectors.quads import (
    build_candidates,
    build_conflict_graph,
    select_polygons,
    select_polygons_ranked,
)


def parse_float_list(raw: str, fallback: Sequence[float]) -> List[float]:
    if not raw:
        return list(dict.fromkeys(float(f"{v:.6f}") for v in fallback))
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def clamp(values: Iterable[float], minimum: float) -> List[float]:
    return [max(minimum, float(v)) for v in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dots", default="dot_results.npz", help="Pre-analysed dot dataset")
    parser.add_argument(
        "--output", default="polygon_results_auto.npz", help="Where to store the tuned polygons"
    )
    parser.add_argument(
        "--selection",
        choices=["graph", "ranked"],
        default="ranked",
        help="Conflict resolution strategy: degree-based graph or score-ranked greedy",
    )
    parser.add_argument(
        "--angle-options",
        default="",
        help="Comma separated overrides for max angle deviation (degrees)",
    )
    parser.add_argument(
        "--side-options",
        default="",
        help="Comma separated overrides for relative side relaxation",
    )
    parser.add_argument(
        "--diag-options",
        default="",
        help="Comma separated overrides for diagonal ratio minimum",
    )
    parser.add_argument(
        "--aspect-options",
        default="",
        help="Comma separated overrides for aspect ratio cap",
    )
    parser.add_argument(
        "--snap-options",
        default="",
        help="Comma separated overrides for snap distance factor",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=60,
        help="Limit the number of configurations evaluated (0 = unlimited)",
    )
    return parser.parse_args()


def generate_overrides(config: DetectionConfig, args: argparse.Namespace) -> List[Dict[str, float]]:
    base = config
    angle_options = clamp(
        parse_float_list(
            args.angle_options,
            [base.max_angle_dev * 0.85, base.max_angle_dev, base.max_angle_dev * 1.15],
        ),
        minimum=5.0,
    )
    side_options = clamp(
        parse_float_list(args.side_options, [base.side_relaxation - 0.1, base.side_relaxation, base.side_relaxation + 0.15]),
        minimum=1.1,
    )
    diag_options = clamp(
        parse_float_list(args.diag_options, [base.diag_ratio_min - 0.03, base.diag_ratio_min, base.diag_ratio_min + 0.05]),
        minimum=1.0,
    )
    aspect_options = clamp(
        parse_float_list(args.aspect_options, [base.aspect_max, base.aspect_max + 0.2]),
        minimum=1.05,
    )
    snap_options = clamp(
        parse_float_list(args.snap_options, [base.distance_snap_factor, base.distance_snap_factor + 0.05]),
        minimum=0.05,
    )

    axis_options = {
        "max_angle_dev": list(dict.fromkeys(angle_options)),
        "side_relaxation": list(dict.fromkeys(side_options)),
        "diag_ratio_min": list(dict.fromkeys(diag_options)),
        "aspect_max": list(dict.fromkeys(aspect_options)),
        "distance_snap_factor": list(dict.fromkeys(snap_options)),
    }

    combos = []
    for values in itertools.product(
        angle_options, side_options, diag_options, aspect_options, snap_options
    ):
        overrides = {
            "max_angle_dev": float(values[0]),
            "side_relaxation": float(values[1]),
            "diag_ratio_min": float(values[2]),
            "aspect_max": float(values[3]),
            "distance_snap_factor": float(values[4]),
        }
        combos.append(overrides)

    # Ensure baseline configuration is evaluated first.
    combos.insert(0, {})

    # Deduplicate while preserving order.
    deduped: List[Dict[str, float]] = []
    seen = set()
    for item in combos:
        key = tuple(sorted(item.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    limit = getattr(args, "max_configs", 0)
    if limit and len(deduped) > limit:
        baseline: Dict[str, float] | None = None
        pool = deduped
        if deduped and deduped[0] == {}:
            baseline = deduped[0]
            pool = deduped[1:]

        trimmed: List[Dict[str, float]] = []
        if baseline is not None:
            trimmed.append(baseline)

        remaining_slots = limit - len(trimmed)
        if remaining_slots <= 0:
            deduped = trimmed[:limit]
            return deduped

        from collections import deque

        angle_buckets: Dict[float, deque[Dict[str, float]]] = {}
        order: List[float] = []
        for entry in pool:
            angle = entry.get("max_angle_dev")
            if angle is None:
                continue
            if angle not in angle_buckets:
                angle_buckets[angle] = deque()
                order.append(angle)
            angle_buckets[angle].append(entry)

        idx = 0
        while len(trimmed) < limit and angle_buckets:
            angle = order[idx % len(order)]
            bucket = angle_buckets.get(angle)
            if not bucket:
                order = [value for value in order if value in angle_buckets and angle_buckets[value]]
                if not order:
                    break
                idx = idx % len(order)
                continue

            trimmed.append(bucket.popleft())
            if not bucket:
                angle_buckets.pop(angle, None)
                order = [value for value in order if value in angle_buckets and angle_buckets[value]]
                if not order:
                    break
                idx = idx % len(order)
                continue

            idx += 1

        # If we still have capacity, fill with any remaining combinations using a stride.
        remaining = [item for bucket in angle_buckets.values() for item in bucket]
        if len(trimmed) < limit and remaining:
            stride = max(1, len(remaining) // max(1, limit - len(trimmed)))
            for i in range(0, len(remaining)):
                if len(trimmed) >= limit:
                    break
                if i % stride == 0:
                    trimmed.append(remaining[i])

        # Guarantee the most permissive angle is represented when possible.
        if angle_options:
            widest_angle = max(angle_options)
            has_widest = any(
                entry.get("max_angle_dev") == widest_angle for entry in trimmed if entry
            )
            if not has_widest:
                candidate = next(
                    (entry for entry in pool if entry.get("max_angle_dev") == widest_angle),
                    None,
                )
                if candidate:
                    if len(trimmed) >= limit:
                        # Replace the last non-baseline entry.
                        for replace_idx in range(len(trimmed) - 1, -1, -1):
                            if trimmed[replace_idx]:
                                trimmed[replace_idx] = candidate
                                break
                    else:
                        trimmed.append(candidate)

        from collections import Counter

        axes = list(axis_options.keys())

        def compute_counts(entries: Sequence[Dict[str, float]]) -> Dict[str, Counter]:
            counts = {axis: Counter() for axis in axes}
            for entry in entries:
                if not entry:
                    for axis in axes:
                        base_value = getattr(base, axis)
                        counts[axis][base_value] += 1
                    continue
                for axis in axes:
                    counts[axis][entry[axis]] += 1
            return counts

        counts = compute_counts(trimmed)

        def missing_values() -> Dict[str, List[float]]:
            return {
                axis: [value for value in axis_options[axis] if counts[axis][value] == 0]
                for axis in axes
            }

        missing = missing_values()
        while any(missing.values()):
            best_entry: Dict[str, float] | None = None
            best_score = 0
            for entry in pool:
                if entry in trimmed:
                    continue
                score = sum(1 for axis in axes if entry[axis] in missing[axis])
                if score > best_score:
                    best_entry = entry
                    best_score = score
            if not best_entry or best_score == 0:
                break

            trimmed.append(best_entry)
            for axis in axes:
                counts[axis][best_entry[axis]] += 1

            if len(trimmed) > limit:
                removable_idx = None
                for idx in range(len(trimmed) - 1, -1, -1):
                    entry = trimmed[idx]
                    if not entry or entry is best_entry:
                        continue
                    can_remove = True
                    for axis in axes:
                        value = entry[axis]
                        if counts[axis][value] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        removable_idx = idx
                        break

                if removable_idx is not None:
                    removed = trimmed.pop(removable_idx)
                    for axis in axes:
                        counts[axis][removed[axis]] -= 1
                else:
                    # Cannot make room without losing coverage; undo the addition.
                    removed = trimmed.pop()
                    for axis in axes:
                        counts[axis][removed[axis]] -= 1
                    break

            missing = missing_values()

        permissive_target = {
            "max_angle_dev": max(axis_options["max_angle_dev"], default=base.max_angle_dev),
            "side_relaxation": max(axis_options["side_relaxation"], default=base.side_relaxation),
            "diag_ratio_min": min(axis_options["diag_ratio_min"], default=base.diag_ratio_min),
            "aspect_max": max(axis_options["aspect_max"], default=base.aspect_max),
            "distance_snap_factor": max(
                axis_options["distance_snap_factor"], default=base.distance_snap_factor
            ),
        }

        target_candidate = next(
            (
                entry
                for entry in pool
                if all(entry.get(axis) == value for axis, value in permissive_target.items())
            ),
            None,
        )
        if target_candidate and target_candidate not in trimmed:
            trimmed.append(target_candidate)
            for axis in axes:
                counts[axis][target_candidate[axis]] += 1

            if len(trimmed) > limit:
                removable_idx = None
                for idx in range(len(trimmed) - 1, -1, -1):
                    entry = trimmed[idx]
                    if not entry or entry is target_candidate:
                        continue
                    can_remove = True
                    for axis in axes:
                        value = entry[axis]
                        if counts[axis][value] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        removable_idx = idx
                        break

                if removable_idx is not None:
                    removed = trimmed.pop(removable_idx)
                    for axis in axes:
                        counts[axis][removed[axis]] -= 1
                else:
                    removed = trimmed.pop()
                    for axis in axes:
                        counts[axis][removed[axis]] -= 1

        deduped = trimmed[:limit]

    return deduped


def evaluate_configuration(
    centers: np.ndarray,
    tasks: Sequence[Tuple[int, int, int]],
    dynamic_side: float,
    base_config: DetectionConfig,
    overrides: Dict[str, float],
    selection: str,
) -> Dict[str, object]:
    config = replace(base_config, **overrides)

    LOGGER.info(
        "Evaluating configuration %s", {k: getattr(config, k) for k in overrides.keys()}
        if overrides
        else {"baseline": True}
    )

    candidates_map = build_candidates(centers, tasks, config, dynamic_side)
    items, neighbors = build_conflict_graph(list(candidates_map.values()), dynamic_side)
    if selection == "ranked":
        chosen = select_polygons_ranked(items, neighbors)
    else:
        chosen = select_polygons(items, neighbors)

    scores = [item["score"] for item in chosen]
    areas = [item["area"] for item in chosen]
    mean_score = float(np.mean(scores)) if scores else float("inf")
    total_area = float(np.sum(areas)) if areas else 0.0

    result = {
        "config": config,
        "overrides": overrides,
        "polygons": [item["points"].copy() for item in chosen],
        "scores": scores,
        "areas": areas,
        "polygon_count": len(chosen),
        "candidate_count": len(candidates_map),
        "mean_score": mean_score,
        "total_area": total_area,
    }
    LOGGER.info(
        " -> %d polygons (candidates: %d, mean score %.2f, total area %.1f)",
        result["polygon_count"],
        result["candidate_count"],
        result["mean_score"],
        result["total_area"],
    )
    return result


def pick_best(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    def score_key(entry: Dict[str, object]):
        return (
            entry["polygon_count"],
            -entry["mean_score"],
            entry["total_area"],
            -entry["candidate_count"],
        )

    best = max(results, key=score_key)
    LOGGER.info(
        "Best configuration selected: %s", best.get("overrides") or {"baseline": True}
    )
    return best


def format_results_table(results: Sequence[Dict[str, object]]) -> str:
    lines = [
        "overrides | polygons | candidates | mean_score | total_area",
        "--------- | -------- | ---------- | ---------- | ----------",
    ]
    for entry in results:
        desc = entry["overrides"] or {"baseline": True}
        lines.append(
            f"{desc} | {entry['polygon_count']} | {entry['candidate_count']} | "
            f"{entry['mean_score']:.2f} | {entry['total_area']:.1f}"
        )
    return "\n".join(lines)


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

    overrides_list = generate_overrides(config, args)
    LOGGER.info("Evaluating %d configuration candidates", len(overrides_list))

    results = [
        evaluate_configuration(centers, tasks, dynamic_side, config, overrides, args.selection)
        for overrides in overrides_list
    ]

    best = pick_best(results)
    table = format_results_table(results)
    LOGGER.info("Search summary:\n%s", table)

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
