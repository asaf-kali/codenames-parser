import cv2
import numpy as np

from codenames_parser.common.debug_util import save_debug_image


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


def normalize(image: np.ndarray, title: str = "normalized") -> np.ndarray:
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # type: ignore[call-overload]
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
