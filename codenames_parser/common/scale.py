import cv2
import numpy as np

from codenames_parser.common.debug_util import save_debug_image


def scale_down_image(image: np.ndarray, max_dimension: int = 800) -> np.ndarray:
    """
    Scale down the image to a maximum dimension of max_dimension.
    """
    height, width = image.shape[:2]
    if max(height, width) <= max_dimension:
        return image
    if height > width:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))
    else:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    small = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    save_debug_image(small, title="scaled_down")
    return small
