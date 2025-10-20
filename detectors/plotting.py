"""Rendering helpers for the detection pipeline."""
from __future__ import annotations

import os
from typing import Sequence

import cv2
import numpy as np
from pytoshop import enums
from pytoshop.user import nested_layers

from . import LOGGER
from .config import DetectionConfig


def cv_to_rgba(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected a BGR image with 3 channels")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGBA).astype(np.uint8)


def rgba_to_layer(name: str, data: np.ndarray) -> nested_layers.Image:
    if data.ndim != 3 or data.shape[2] != 4:
        raise ValueError(f"Layer {name} is not RGBA")
    if data.dtype != np.uint8:
        raise ValueError(f"Layer {name} must be uint8")

    height, width = data.shape[:2]
    planes = np.transpose(data, (2, 0, 1))
    channels = {
        0: planes[0],
        1: planes[1],
        2: planes[2],
        enums.ChannelId.transparency: planes[3],
    }

    return nested_layers.Image(
        name=name,
        top=0,
        left=0,
        bottom=height,
        right=width,
        color_mode=enums.ColorMode.rgb,
        channels=channels,
    )


def render_layers(
    image: np.ndarray,
    centers: np.ndarray,
    polygons: Sequence[np.ndarray],
    config: DetectionConfig,
    output_path: str,
    export_psd: bool = True,
) -> str:
    """Render the detections to disk and optionally save a layered PSD."""

    LOGGER.info("Rendering output layers")
    lines = np.zeros_like(image)
    for poly in polygons:
        cv2.polylines(lines, [poly.astype(np.int32)], True, config.polygon_color_bgr, config.line_thickness)
    dots = np.zeros_like(image)
    for x, y in centers:
        cv2.circle(dots, (int(x), int(y)), config.dot_radius, config.dot_color_bgr, -1)

    final = image.copy()
    final = cv2.add(final, lines)
    final = cv2.add(final, dots)

    cv2.imwrite(output_path, final)
    LOGGER.info("Saved combined output to %s", output_path)

    if not export_psd:
        return output_path

    background = cv_to_rgba(image)
    lines_layer = cv2.cvtColor(lines, cv2.COLOR_BGR2RGBA).astype(np.uint8)
    dots_layer = cv2.cvtColor(dots, cv2.COLOR_BGR2RGBA).astype(np.uint8)

    layers = [
        rgba_to_layer("Background", background),
        rgba_to_layer("Lines", lines_layer),
        rgba_to_layer("Dots", dots_layer),
    ]

    height, width = image.shape[:2]
    psd = nested_layers.nested_layers_to_psd(
        layers,
        color_mode=enums.ColorMode.rgb,
        depth=8,
        size=(height, width),
        compression=enums.Compression.raw,
    )
    psd_path = output_path.replace(".png", "_layers.psd")
    try:
        with open(psd_path, "wb") as handle:
            psd.write(handle)
        if not os.path.exists(psd_path) or os.path.getsize(psd_path) < 1024:
            raise IOError("PSD export produced an empty file")
        LOGGER.info("Saved layered PSD to %s", psd_path)
    except Exception as exc:  # pragma: no cover - logging side-effect only
        LOGGER.warning("PSD export failed: %s", exc)

    return output_path


__all__ = ["render_layers"]
