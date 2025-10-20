"""Stage 1: detect dots and pre-compute neighbourhood tasks."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from detectors import LOGGER
from detectors.config import DetectionConfig
from detectors.dots import detect_dots, load_image, preanalyse_dots


def save_dot_payload(path: Path, image_path: str, image_shape, payload: Dict[str, np.ndarray]) -> None:
    """Persist the dot detection output for later reuse."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image_path=np.array(image_path),
        image_height=np.array(image_shape[0], dtype=np.int32),
        image_width=np.array(image_shape[1], dtype=np.int32),
        **payload,
    )
    LOGGER.info("Dot data saved to %s (%.2f KB)", path, path.stat().st_size / 1024.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="back_scheme.png", help="Input image path")
    parser.add_argument(
        "--output",
        default="dot_results.npz",
        help="Path to store the detected centers and pre-analysis data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DetectionConfig(image_path=args.image, dot_data_path=args.output)

    image = load_image(config.image_path)
    centers = detect_dots(image, config)
    if len(centers) < 4:
        raise RuntimeError("Insufficient dots detected")

    payload = preanalyse_dots(centers, config)
    LOGGER.info(
        "Pre-analysis: dynamic side %.2f, angle tolerance %.2f (estimated %.2f)",
        float(payload["dynamic_side"]),
        float(payload["angle_tolerance"]),
        float(payload["estimated_side"]),
    )
    save_dot_payload(Path(config.dot_data_path), config.image_path, image.shape[:2], payload)


if __name__ == "__main__":
    main()
