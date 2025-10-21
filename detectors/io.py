"""Utility helpers for loading and saving stage datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from . import LOGGER


def load_dot_data(path: Path) -> Dict[str, np.ndarray]:
    """Load the cached dot detection payload from ``path``."""

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


def _metadata_to_arrays(metadata: Dict[str, object]) -> Dict[str, np.ndarray]:
    """Convert metadata entries into ``np.ndarray`` payload fields."""

    output: Dict[str, np.ndarray] = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            output[key] = value
        elif isinstance(value, (bytes, bytearray)):
            output[key] = np.frombuffer(value, dtype=np.uint8)
        elif isinstance(value, (float, int)):
            output[key] = np.array(value)
        elif isinstance(value, (list, tuple)):
            output[key] = np.array(value)
        else:
            output[key] = np.array(str(value))
    return output


def save_polygon_payload(
    path: Path,
    centers: np.ndarray,
    polygons: Iterable[np.ndarray],
    scores: Iterable[float],
    metadata: Dict[str, object],
) -> None:
    """Persist polygon detection results alongside auxiliary metadata."""

    polygon_list = [np.asarray(poly, dtype=np.float32) for poly in polygons]
    if polygon_list:
        polygon_array = np.stack(polygon_list, axis=0)
    else:
        polygon_array = np.empty((0, 0, 2), dtype=np.float32)

    payload = {
        "centers": centers.astype(np.float32),
        "polygons": polygon_array,
        "scores": np.asarray(list(scores), dtype=np.float32),
    }
    payload.update(
        _metadata_to_arrays(
            {
                "image_path": metadata.get("image_path", ""),
                "image_height": metadata.get("image_height", 0),
                "image_width": metadata.get("image_width", 0),
                "dynamic_side": metadata.get("dynamic_side", 0.0),
                "angle_tolerance": metadata.get("angle_tolerance", 0.0),
                "estimated_side": metadata.get("estimated_side", 0.0),
                "spread": metadata.get("spread", 0.0),
            }
        )
    )

    # Persist any additional metadata fields beyond the standard keys.
    standard_keys = set(payload.keys())
    for key, value in metadata.items():
        if key in standard_keys:
            continue
        payload[key] = _metadata_to_arrays({key: value})[key]

    np.savez_compressed(path, **payload)
    LOGGER.info("Polygon data saved to %s (%.2f KB)", path, path.stat().st_size / 1024.0)


__all__ = ["load_dot_data", "save_polygon_payload"]
