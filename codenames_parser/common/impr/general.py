import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


def sharpen(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened
