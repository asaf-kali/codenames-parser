from dataclasses import dataclass

import cv2
import numpy as np

from codenames_parser.common.debug_util import save_debug_image


@dataclass
class ScaleResult:
    image: np.ndarray
    scale_factor: float


def scale_down_image(image: np.ndarray, max_dimension: int = 800) -> ScaleResult:
    """
    Scale down the image to a maximum dimension of max_dimension.
    """
    height, width = image.shape[:2]
    if max(height, width) <= max_dimension:
        return ScaleResult(image=image, scale_factor=1.0)
    if height > width:
        factor = max_dimension / height
        new_width = int(width * factor)
        new_height = max_dimension
    else:
        factor = max_dimension / width
        new_width = max_dimension
        new_height = int(height * factor)
    scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    save_debug_image(scaled, title="scaled down")
    return ScaleResult(image=scaled, scale_factor=factor)


def downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    """Downsample the image by the given factor.

    Args:
        image (np.ndarray): Input image.
        factor (int): Downsampling factor.

    Returns:
        np.ndarray: Downsampled image.
    """
    if factor == 1:
        return image
    height, width = image.shape[:2]
    new_size = (width // factor, height // factor)
    downsampled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return downsampled_image
