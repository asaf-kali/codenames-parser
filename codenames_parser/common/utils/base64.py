import base64

import cv2
import numpy as np


def base64_to_image(b64_string: str) -> np.ndarray:
    # Decode base64 string to bytes
    image_data = base64.b64decode(b64_string)
    # Convert bytes to a numpy array
    np_array = np.frombuffer(image_data, np.uint8)
    # Decode numpy array to OpenCV image format (BGR)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image


def image_to_base64(image: np.ndarray) -> str:
    # Encode the image as a JPEG to a memory buffer
    _, buffer = cv2.imencode(".jpg", image)
    # Convert the buffer to a base64 string
    b64_string = base64.b64encode(buffer).decode("utf-8")
    return b64_string
