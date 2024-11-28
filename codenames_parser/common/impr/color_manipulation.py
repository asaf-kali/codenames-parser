import logging

import cv2
import numpy as np

from codenames_parser.common.utils.debug_util import save_debug_image

log = logging.getLogger(__name__)


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if not is_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def is_grayscale(image: np.ndarray) -> bool:
    return len(image.shape) == 2 or image.shape[2] == 1


def normalize(image: np.ndarray, title: str = "normalized", save: bool = False) -> np.ndarray:
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # type: ignore[call-overload]
    if save:
        save_debug_image(normalized, title=title)
    return normalized


def quantize(image: np.ndarray, k: int = 10) -> np.ndarray:
    log.debug(f"Quantizing image with k={k}")
    image = image.copy()
    reshape = (-1, 1) if is_grayscale(image) else (-1, 3)
    z = image.reshape(reshape).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centers = cv2.kmeans(
        z, K=k, bestLabels=None, criteria=criteria, attempts=10, flags=flags
    )  # type: ignore
    centers = np.uint8(centers)
    quantized_image = centers[labels.flatten()]
    quantized_image = quantized_image.reshape(image.shape)
    save_debug_image(quantized_image, title="quantized")
    return quantized_image
