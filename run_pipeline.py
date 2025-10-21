"""Utility script to run the dot and polygon detection pipeline end-to-end."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> None:
    """Execute *command* and raise if it fails."""

    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        help="Path to the input image that will be processed by the pipeline",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Base directory where pipeline scripts live (default: script directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    base_dir = args.working_dir.resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Working directory not found: {base_dir}")

    scripts_dir = base_dir

    image_stem = image_path.stem
    image_suffix = image_path.suffix or ".png"
    output_dir = image_path.parent / image_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    dots_npz = output_dir / f"{image_stem}_dots.npz"
    polygons_npz = output_dir / f"{image_stem}_polygons.npz"
    dots_image = output_dir / f"{image_stem}_dots{image_suffix}"
    polygons_image = output_dir / f"{image_stem}_polygons{image_suffix}"

    python_executable = sys.executable

    run_command(
        [
            python_executable,
            str(scripts_dir / "dots_detection.py"),
            "--image",
            str(image_path),
            "--output",
            str(dots_npz),
        ]
    )

    run_command(
        [
            python_executable,
            str(scripts_dir / "plot_detections.py"),
            "--dots",
            str(dots_npz),
            "--no-polygons",
            "--no-psd",
            "--output",
            str(dots_image),
        ]
    )

    run_command(
        [
            python_executable,
            str(scripts_dir / "rectangles_autotune_opt.py"),
            "--dots",
            str(dots_npz),
            "--output",
            str(polygons_npz),
        ]
    )

    run_command(
        [
            python_executable,
            str(scripts_dir / "plot_detections.py"),
            "--polygons",
            str(polygons_npz),
            "--dots",
            str(dots_npz),
            "--no-psd",
            "--output",
            str(polygons_image),
        ]
    )

    root_results = scripts_dir / f"{image_stem}_results{image_suffix}"
    shutil.copy2(polygons_image, root_results)
    print(f"Copied {polygons_image} to {root_results}")


if __name__ == "__main__":
    main()
