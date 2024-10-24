import cv2
import numpy as np


def transform(image: np.ndarray, scale: float, angle: float) -> np.ndarray:
    # Get the image dimensions
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    # Compute the rotation matrix with scaling
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # Compute the sine and cosine of the rotation angle
    angle_rad = np.deg2rad(angle)
    abs_cos = abs(np.cos(angle_rad) * scale)
    abs_sin = abs(np.sin(angle_rad) * scale)

    # Compute the new bounding dimensions
    bound_w = int(width * abs_cos + height * abs_sin)
    bound_h = int(width * abs_sin + height * abs_cos)

    # Adjust the rotation matrix to account for translation
    rotation_matrix[0, 2] += (bound_w / 2) - center[0]
    rotation_matrix[1, 2] += (bound_h / 2) - center[1]

    # Perform the rotation and scaling
    transformed = cv2.warpAffine(
        image,
        rotation_matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return transformed
