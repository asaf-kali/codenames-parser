from typing import NamedTuple

import cv2
import numpy as np

from codenames_parser.common.utils.debug_util import save_debug_image
from codenames_parser.common.utils.models import Color


class ColorDistanceResult(NamedTuple):
    mask: np.ndarray
    filtered_negative: np.ndarray


def color_distance_negative(image: np.ndarray, color: Color) -> np.ndarray:
    """
    Calculates the negative of the color distance.
    i.e., the closer the pixel is to the color, the higher the value.
    """
    norms = np.linalg.norm(image - color.vector, axis=2)
    # Normalize the distance
    max_distance = np.max(norms)
    normalized = norms / max_distance
    negative = 1 - normalized
    return negative


def color_distance_mask(image: np.ndarray, color: Color, percentile: int = 80) -> ColorDistanceResult:
    negative = color_distance_negative(image, color=color)
    negative_image = (255 * negative).astype(np.uint8)
    # save_debug_image(negative_image, title=f"normalized for {color}")
    equalized = cv2.equalizeHist(negative_image)
    save_debug_image(equalized, title=f"equalized for {color}")
    threshold = np.percentile(negative, q=percentile)
    mask = negative > threshold
    filtered = apply_mask(equalized, mask=mask)
    save_debug_image(filtered, title=f"threshold for {color}")
    return ColorDistanceResult(mask=mask, filtered_negative=filtered)


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies a mask to an image.
    """
    if len(image.shape) == 3:
        mask = np.stack([mask] * 3, axis=-1).astype(image.dtype)
    else:
        mask = mask.astype(image.dtype)
    return cv2.multiply(image, mask)
