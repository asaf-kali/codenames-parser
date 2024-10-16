import logging

import cv2
import numpy as np

from codenames_parser.common.debug_util import SEPARATOR, save_debug_image

log = logging.getLogger(__name__)


def read_image(image_path: str) -> np.ndarray:
    """
    Reads an image from a file.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: The image in BGR format.
    """
    log.info(SEPARATOR)
    log.info(f"Reading image from {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")
    save_debug_image(image, title="input")
    return image