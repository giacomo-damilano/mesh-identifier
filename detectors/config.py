"""Configuration dataclasses shared across the detection pipeline."""
from dataclasses import dataclass, field
import multiprocessing
from typing import Tuple


@dataclass
class DetectionConfig:
    """Runtime configuration shared between the detection stages."""

    image_path: str = "back_scheme.png"
    dot_data_path: str = "dot_results.npz"
    polygon_data_path: str = "polygon_results.npz"
    output_path: str = "back_scheme_detected.png"

    dot_radius: int = 6
    line_thickness: int = 2
    dot_color_bgr: Tuple[int, int, int] = (0, 0, 255)
    polygon_color_bgr: Tuple[int, int, int] = (0, 255, 0)

    preferred_side: float = 150.0
    preferred_angle: float = 90.0
    max_triplets_per_batch: int = 160
    min_area_ratio: float = 0.015
    max_angle_dev: float = 38.0
    side_relaxation: float = 1.55
    diag_ratio_min: float = 1.05
    aspect_max: float = 1.45
    distance_snap_factor: float = 0.2

    max_workers: int = field(
        default_factory=lambda: max(1, multiprocessing.cpu_count() - 1)
    )


__all__ = ["DetectionConfig"]
