"""Stage 3: visualise polygon detections on top of the source image."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from detectors import LOGGER
from detectors.config import DetectionConfig
from detectors.dots import load_image
from detectors.plotting import render_layers


def load_polygon_data(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    raw_image_path = data["image_path"]
    if getattr(raw_image_path, "shape", ()) == ():
        image_path = str(raw_image_path.item())
    else:
        image_path = str(raw_image_path)
    return {
        "centers": data["centers"].astype(np.float32),
        "polygons": data["polygons"].astype(np.float32),
        "scores": data["scores"].astype(np.float32),
        "image_path": image_path,
        "image_height": int(data["image_height"]),
        "image_width": int(data["image_width"]),
        "dynamic_side": float(data["dynamic_side"]),
        "angle_tolerance": float(data["angle_tolerance"]),
        "spread": float(data["spread"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polygons", default="polygon_results.npz", help="Polygon dataset")
    parser.add_argument("--image", default=None, help="Optional override for the source image path")
    parser.add_argument("--output", default="back_scheme_detected.png", help="Rendered image path")
    parser.add_argument("--no-psd", action="store_true", help="Disable layered PSD export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    polygon_payload = load_polygon_data(Path(args.polygons))

    image_path = args.image or polygon_payload["image_path"]
    config = DetectionConfig(
        image_path=image_path,
        output_path=args.output,
    )

    image = load_image(config.image_path)
    centers = polygon_payload["centers"]
    polygons_array = polygon_payload["polygons"]
    polygons = [poly for poly in polygons_array]
    LOGGER.info(
        "Rendering %d polygons on %s", len(polygons), config.image_path
    )
    render_layers(
        image,
        centers,
        polygons,
        config,
        config.output_path,
        export_psd=not args.no_psd,
    )


if __name__ == "__main__":
    main()
