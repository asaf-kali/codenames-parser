from dataclasses import dataclass

import cv2
import numpy as np


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
        scale_factor = max_dimension / height
        resized = resize_image(image, dst_height=max_dimension)
    else:
        scale_factor = max_dimension / width
        resized = resize_image(image, dst_width=max_dimension)
    return ScaleResult(image=resized, scale_factor=scale_factor)


def resize_image(image: np.ndarray, dst_width: int | None = None, dst_height: int | None = None) -> np.ndarray:
    if not dst_width and not dst_height:
        raise ValueError("At least one of dst_width or dst_height must be provided.")
    height, width = image.shape[:2]
    if dst_width:
        factor = dst_width / width
        dst_height = int(height * factor)
    else:
        factor = dst_height / height
        dst_width = int(width * factor)
    resized = cv2.resize(image, dsize=(dst_width, dst_height), interpolation=cv2.INTER_AREA)  # type: ignore
    return resized


def downsample_image(image: np.ndarray, factor: float) -> np.ndarray:
    """Downsample the image by the given factor.

    Args:
        image (np.ndarray): Input image.
        factor (float): Downsampling factor.

    Returns:
        np.ndarray: Downsampled image.
    """
    if factor == 1:
        return image
    _, width = image.shape[:2]
    dst_width = width // factor
    return resize_image(image, dst_width=dst_width)


def has_larger_dimension(image: np.ndarray, other: np.ndarray) -> bool:
    return image.shape[0] > other.shape[0] or image.shape[1] > other.shape[1]
