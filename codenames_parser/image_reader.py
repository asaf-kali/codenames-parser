import cv2
import numpy as np

from codenames_parser.debugging.util import save_debug_image


def read_image(image_path: str) -> np.ndarray:
    """
    Reads an image from a file.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: The image in BGR format.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")
    save_debug_image(image, title="input")
    return image
