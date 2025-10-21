"""Apply PCA-based preprocessing to enhance dot detection."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from detectors import LOGGER
from detectors.config import DetectionConfig
from detectors.dots import detect_dots, load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="back_scheme.png", help="Input image path")
    parser.add_argument(
        "--pca-image",
        default="back_scheme_pca.png",
        help="Path to write the first principal component reconstruction",
    )
    parser.add_argument(
        "--component-image",
        default=None,
        help="Optional path to store the normalised first principal component (single channel)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run dot detection on both the original and PCA images for comparison",
    )
    return parser.parse_args()


def compute_first_component(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the rank-1 reconstruction and normalised first component.

    The function computes a PCA over the colour channels treating each pixel as a
    three-dimensional vector. It returns a rank-1 reconstruction that retains the
    global colour balance together with a single-channel representation of the
    first principal component scores.
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR image with three channels")

    h, w, c = image.shape
    flat = image.reshape(-1, c)
    num_pixels = flat.shape[0]
    LOGGER.info("Running PCA on %d pixels", num_pixels)

    mean = flat.mean(axis=0).astype(np.float32)
    if int(image.min()) == int(image.max()):
        LOGGER.warning("Image has no variance; returning the original image")
        component = np.zeros((h, w), dtype=np.uint8)
        return image.copy(), component

    chunk_size = 1_000_000
    cov = np.zeros((c, c), dtype=np.float64)
    for start in range(0, num_pixels, chunk_size):
        end = min(start + chunk_size, num_pixels)
        chunk = flat[start:end].astype(np.float32)
        diff = chunk - mean
        cov += diff.T @ diff

    if num_pixels > 1:
        cov /= (num_pixels - 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    if eigvals.size == 0 or not np.isfinite(eigvals[-1]):
        LOGGER.warning("Unable to compute a meaningful principal component")
        component = np.zeros((h, w), dtype=np.uint8)
        return image.copy(), component

    principal = eigvecs[:, np.argmax(eigvals)].astype(np.float32)
    norm = float(np.linalg.norm(principal))
    if norm <= 0:
        LOGGER.warning("Principal component has zero norm")
        component = np.zeros((h, w), dtype=np.uint8)
        return image.copy(), component
    principal /= norm

    rank1_image = np.empty_like(image)
    rank1_flat = rank1_image.reshape(-1, c)

    min_score = float("inf")
    max_score = float("-inf")
    for start in range(0, num_pixels, chunk_size):
        end = min(start + chunk_size, num_pixels)
        chunk = flat[start:end].astype(np.float32)
        diff = chunk - mean
        scores = diff @ principal
        min_score = min(min_score, float(scores.min()))
        max_score = max(max_score, float(scores.max()))
        recon = mean + np.outer(scores, principal)
        rank1_flat[start:end] = np.clip(recon, 0, 255).astype(np.uint8)

    component_vis = np.zeros((h, w), dtype=np.uint8)
    if max_score > min_score:
        component_flat = component_vis.reshape(-1)
        scale = max_score - min_score
        for start in range(0, num_pixels, chunk_size):
            end = min(start + chunk_size, num_pixels)
            chunk = flat[start:end].astype(np.float32)
            diff = chunk - mean
            scores = diff @ principal
            normalised = (scores - min_score) / scale
            component_flat[start:end] = np.clip(normalised * 255.0, 0, 255).astype(np.uint8)

    return rank1_image, component_vis


def save_image(path: str | None, image: np.ndarray) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cv2.imwrite(str(output_path), image):
        LOGGER.info("Saved image to %s", output_path)
    else:
        LOGGER.error("Failed to write image to %s", output_path)


def compare_detections(original: np.ndarray, processed: np.ndarray, config: DetectionConfig) -> None:
    LOGGER.info("Running dot detection on the original image")
    original_centers = detect_dots(original, config)
    LOGGER.info("Original image: detected %d dots", len(original_centers))

    LOGGER.info("Running dot detection on the PCA-processed image")
    processed_centers = detect_dots(processed, config)
    LOGGER.info("PCA image: detected %d dots", len(processed_centers))

    if len(processed_centers) >= len(original_centers):
        LOGGER.info("PCA preprocessing did not reduce the number of detected dots")
    else:
        LOGGER.info("PCA preprocessing reduced the number of detected dots")



def main() -> None:
    args = parse_args()
    config = DetectionConfig(image_path=args.image)

    original_image = load_image(config.image_path)
    pca_image, component = compute_first_component(original_image)

    save_image(args.pca_image, pca_image)
    if args.component_image:
        # Store the component as a grayscale PNG (single channel).
        save_image(args.component_image, component)

    if args.compare:
        compare_detections(original_image, pca_image, config)


if __name__ == "__main__":
    main()
