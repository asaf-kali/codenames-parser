import cv2
import numpy as np

from codenames_parser.debugging.util import save_debug_image

MAX_DIMENSION = 800


def scale_down_image(image: np.ndarray) -> np.ndarray:
    """
    Scale down the image to a maximum dimension of MAX_DIMENSION.
    """
    height, width = image.shape[:2]
    if max(height, width) <= MAX_DIMENSION:
        return image
    if height > width:
        new_height = MAX_DIMENSION
        new_width = int(width * (MAX_DIMENSION / height))
    else:
        new_width = MAX_DIMENSION
        new_height = int(height * (MAX_DIMENSION / width))
    small = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    save_debug_image(small, title="scaled_down")
    return small
