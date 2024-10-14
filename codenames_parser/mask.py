import cv2
import numpy as np

from codenames_parser.debugging.util import save_debug_image
from codenames_parser.models import Color


def color_distance_mask(image: np.ndarray, color: Color) -> np.ndarray:
    """
    Calculates the Euclidean distance between the image and a color.
    """
    norms = np.linalg.norm(image - color.vector, axis=2)
    # Normalize the distance
    max_distance = np.max(norms)
    normalized = norms / max_distance
    # save_debug_image((normalized * 255).astype(np.uint8), title=f"normalized for {color}")
    # Histogram equalization
    # equalized = cv2.equalizeHist((normalized * 255).astype(np.uint8))
    # save_debug_image(equalized, title=f"equalized for {color}")
    # Take only top 30% of pixels
    threshold = np.percentile(normalized, 70)
    mask = normalized > threshold
    filtered = cv2.multiply(image, np.stack([mask] * 3, axis=-1).astype(np.uint8))
    save_debug_image(filtered, title=f"threshold for {color}")
    return mask


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies a mask to an image.
    """
    if len(image.shape) == 3:
        mask = np.stack([mask] * 3, axis=-1).astype(np.uint8)
    return cv2.multiply(image, mask)
