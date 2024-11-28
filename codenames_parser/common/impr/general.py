import logging
from typing import Callable

import cv2
import numpy as np

from codenames_parser.common.impr.color_manipulation import is_grayscale

log = logging.getLogger(__name__)


def sharpen(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened


def apply_per_channel(image: np.ndarray, func: Callable, *args, **kwargs) -> np.ndarray:
    if is_grayscale(image):
        return func(image, *args, **kwargs)
    channels = cv2.split(image)
    channels_processed = [func(channel, *args, **kwargs) for channel in channels]
    colored_processed = cv2.merge(channels_processed)
    return colored_processed
