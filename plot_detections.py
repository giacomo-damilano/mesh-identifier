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
    with np.load(path, allow_pickle=True) as data:
        raw_image_path = data["image_path"]
        if getattr(raw_image_path, "shape", ()) == ():
            image_path = str(raw_image_path.item())
        else:
            image_path = str(raw_image_path)

        payload = {
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

    return payload


def load_dot_data(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        raw_image_path = data["image_path"]
        if getattr(raw_image_path, "shape", ()) == ():
            image_path = str(raw_image_path.item())
        else:
            image_path = str(raw_image_path)

        payload: Dict[str, np.ndarray] = {
            "centers": data["centers"].astype(np.float32),
            "image_path": image_path,
            "image_height": int(data["image_height"]),
            "image_width": int(data["image_width"]),
        }

        for optional_key in (
            "dynamic_side",
            "spread",
            "angle_tolerance",
            "estimated_side",
            "tasks",
        ):
            if optional_key in data.files:
                payload[optional_key] = data[optional_key]

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--polygons",
        default=None,
        help="Polygon dataset (default: polygon_results.npz)",
    )
    parser.add_argument(
        "--no-polygons",
        action="store_true",
        help="Disable polygon overlay even if data is available",
    )
    parser.add_argument(
        "--dots",
        default=None,
        help="Dot dataset to render detected centers",
    )
    parser.add_argument("--image", default=None, help="Optional override for the source image path")
    parser.add_argument("--output", default="back_scheme_detected.png", help="Rendered image path")
    parser.add_argument("--no-psd", action="store_true", help="Disable layered PSD export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = DetectionConfig()

    dot_payload: Dict[str, np.ndarray] | None = None
    if args.dots is not None:
        dot_payload = load_dot_data(Path(args.dots))

    polygon_payload: Dict[str, np.ndarray] | None = None
    if not args.no_polygons:
        polygon_path = Path(
            args.polygons if args.polygons is not None else base_config.polygon_data_path
        )
        if polygon_path.exists():
            polygon_payload = load_polygon_data(polygon_path)
        elif args.polygons is not None:
            raise FileNotFoundError(f"Polygon dataset not found: {polygon_path}")
        elif dot_payload is None:
            raise FileNotFoundError(
                f"Polygon dataset not found: {polygon_path}. Provide --no-polygons or "
                "a valid --polygons path."
            )
        else:
            LOGGER.info("Polygon dataset %s not found; rendering dots only", polygon_path)

    if polygon_payload is None and dot_payload is None:
        raise RuntimeError("No polygon or dot data available to render")

    image_path = args.image
    if not image_path:
        if polygon_payload is not None:
            image_path = polygon_payload["image_path"]
        elif dot_payload is not None:
            image_path = dot_payload["image_path"]

    if not image_path:
        raise RuntimeError("Unable to determine source image path")

    config = DetectionConfig(
        image_path=image_path,
        output_path=args.output,
    )

    image = load_image(config.image_path)

    if dot_payload is not None:
        centers = dot_payload["centers"]
        LOGGER.info("Rendering %d dots from %s", len(centers), args.dots)
    else:
        centers = polygon_payload["centers"]  # type: ignore[index]

    polygons = []
    if polygon_payload is not None:
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

