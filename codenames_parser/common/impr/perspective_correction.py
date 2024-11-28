import cv2
import numpy as np

from codenames_parser.common.utils.debug_util import save_debug_image


def correct_perspective(image: np.ndarray) -> np.ndarray:
    """
    Corrects the perspective of the image to get a top-down view of the board.

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The perspective-corrected image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        # If the approximated contour has 4 points, we assume it's the board
        if len(approx) == 4:
            board_contour = approx
            break
    else:
        raise ValueError("Board contour not found")

    # Get the top-down view of the board
    warped = four_point_transform(image, board_contour.reshape(4, 2))
    save_debug_image(warped, title="Perspective Correction")
    return warped


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders points in the order: top-left, top-right, bottom-right, bottom-left.

    Args:
        pts (np.ndarray): An array of four points.

    Returns:
        np.ndarray: The ordered points.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum and difference to find corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Performs a perspective transform to obtain a top-down view.

    Args:
        image (np.ndarray): The input image.
        pts (np.ndarray): The corner points of the board.

    Returns:
        np.ndarray: The warped image.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width and height
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    # Destination points
    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")

    # Perspective transform
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height))

    return warped
