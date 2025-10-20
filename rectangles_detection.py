"""Stage 2: identify four-sided polygons from pre-analysed dots."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from detectors import LOGGER
from detectors.config import DetectionConfig
from detectors.quads import build_candidates, build_conflict_graph, select_polygons


def load_dot_data(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    raw_image_path = data["image_path"]
    if getattr(raw_image_path, "shape", ()) == ():
        image_path = str(raw_image_path.item())
    else:
        image_path = str(raw_image_path)
    payload = {
        "centers": data["centers"].astype(np.float32),
        "tasks": data["tasks"].astype(np.int32),
        "dynamic_side": float(data["dynamic_side"]),
        "spread": float(data["spread"]),
        "angle_tolerance": float(data["angle_tolerance"]),
        "estimated_side": float(data["estimated_side"]),
        "image_path": image_path,
        "image_height": int(data["image_height"]),
        "image_width": int(data["image_width"]),
    }
    return payload


def save_polygon_payload(
    path: Path,
    centers: np.ndarray,
    polygons: List[np.ndarray],
    scores: List[float],
    metadata: Dict[str, float],
) -> None:
    if polygons:
        polygon_array = np.stack([poly.astype(np.float32) for poly in polygons], axis=0)
    else:
        polygon_array = np.empty((0, 0, 2), dtype=np.float32)
    np.savez_compressed(
        path,
        centers=centers.astype(np.float32),
        polygons=polygon_array,
        scores=np.array(scores, dtype=np.float32),
        image_path=np.array(metadata["image_path"]),
        image_height=np.array(metadata["image_height"], dtype=np.int32),
        image_width=np.array(metadata["image_width"], dtype=np.int32),
        dynamic_side=np.array(metadata["dynamic_side"], dtype=np.float32),
        angle_tolerance=np.array(metadata["angle_tolerance"], dtype=np.float32),
        estimated_side=np.array(metadata["estimated_side"], dtype=np.float32),
        spread=np.array(metadata["spread"], dtype=np.float32),
    )
    LOGGER.info("Polygon data saved to %s (%.2f KB)", path, path.stat().st_size / 1024.0)


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
