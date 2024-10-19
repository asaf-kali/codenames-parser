import cv2
import numpy as np


def has_larger_dimension(image: np.ndarray, other: np.ndarray) -> bool:
    return image.shape[0] > other.shape[0] or image.shape[1] > other.shape[1]


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def zero_pad(image: np.ndarray, padding: int) -> np.ndarray:
    """Pad the image with zeros on all sides.

    Args:
        image (np.ndarray): Input image.
        padding (int): Padding size.

    Returns:
        np.ndarray: Padded image.
    """
    p = padding
    return cv2.copyMakeBorder(image, p, p, p, p, cv2.BORDER_CONSTANT, value=0)  # type: ignore


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
