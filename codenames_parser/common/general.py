import logging

import cv2
import numpy as np

from codenames_parser.common.debug_util import save_debug_image

log = logging.getLogger(__name__)


def has_larger_dimension(image: np.ndarray, other: np.ndarray) -> bool:
    return image.shape[0] > other.shape[0] or image.shape[1] > other.shape[1]


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    equalized = cv2.equalizeHist(image)
    save_debug_image(equalized, title="equalized")
    return equalized


def normalize(image: np.ndarray, title: str = "normalized", save: bool = False) -> np.ndarray:
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # type: ignore[call-overload]
    if save:
        save_debug_image(normalized, title=title)
    return normalized


def value_pad(image: np.ndarray, padding: int, value: int) -> np.ndarray:
    """Pad the image with a constant value on all sides.

    Args:
        image (np.ndarray): Input image.
        padding (int): Padding size.
        value (int): Padding value.

    Returns:
        np.ndarray: Padded image.
    """
    p = padding
    return cv2.copyMakeBorder(image, p, p, p, p, cv2.BORDER_CONSTANT, value=value)  # type: ignore


def zero_pad(image: np.ndarray, padding: int) -> np.ndarray:
    return value_pad(image, padding, value=0)


def border_pad(image: np.ndarray, padding: int) -> np.ndarray:
    """Pad the image with the value of the closest border pixel.

    Args:
        image (np.ndarray): Input image.
        padding (int): Padding size.

    Returns:
        np.ndarray: Padded image.
    """
    p = padding
    return cv2.copyMakeBorder(image, p, p, p, p, cv2.BORDER_REPLICATE)


def quantize(image: np.ndarray, k: int = 10) -> np.ndarray:
    log.debug(f"Quantizing image with k={k}")
    image = image.copy()
    reshape = (-1, 1) if _is_grayscale(image) else (-1, 3)
    z = image.reshape(reshape).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centers = cv2.kmeans(
        z, K=k, bestLabels=None, criteria=criteria, attempts=10, flags=flags
    )  # type: ignore
    centers = np.uint8(centers)
    quantized_image = centers[labels.flatten()]
    quantized_image = quantized_image.reshape(image.shape)
    return quantized_image


def _is_grayscale(image: np.ndarray) -> bool:
    return len(image.shape) == 2 or image.shape[2] == 1


def sharpen(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened
