import cv2
import numpy as np


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
