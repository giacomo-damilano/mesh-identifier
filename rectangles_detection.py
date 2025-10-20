"""Stage 2: identify four-sided polygons from pre-analysed dots."""
from __future__ import annotations

import argparse
from pathlib import Path

from detectors import LOGGER
from detectors.config import DetectionConfig
from detectors.io import load_dot_data, save_polygon_payload
from detectors.quads import build_candidates, build_conflict_graph, select_polygons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dots", default="dot_results.npz", help="Pre-analysed dot dataset")
    parser.add_argument(
        "--output",
        default="polygon_results.npz",
        help="Path to store the detected polygons",
    )
    return parser.parse_args()


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

    candidates_map = build_candidates(centers, tasks, config, dynamic_side)
    items, neighbors = build_conflict_graph(list(candidates_map.values()), dynamic_side)
    selected = select_polygons(items, neighbors)
    LOGGER.info("Final polygon count: %d", len(selected))

    polygons = [item["points"] for item in selected]
    scores = [item["score"] for item in selected]
    metadata = {
        "image_path": dot_payload["image_path"],
        "image_height": dot_payload["image_height"],
        "image_width": dot_payload["image_width"],
        "dynamic_side": dynamic_side,
        "angle_tolerance": config.max_angle_dev,
        "estimated_side": dot_payload["estimated_side"],
        "spread": spread,
    }
    save_polygon_payload(Path(config.polygon_data_path), centers, polygons, scores, metadata)


if __name__ == "__main__":
    main()
