import logging
from typing import NamedTuple

import cv2
import numpy as np

from codenames_parser.debugging.util import draw_lines, save_debug_image
from codenames_parser.models import GridLines, Line

log = logging.getLogger(__name__)


def align_image(image: np.ndarray) -> np.ndarray:
    rho = 1
    while True:
        result = _align_image_iteration(image, rho=rho)
        if result.line_count < 10:
            break
        if abs(result.angle_degrees) < 0.01:
            break
        image = result.aligned_image
        rho /= 1.5
    return image


class AlignmentIterationResult(NamedTuple):
    aligned_image: np.ndarray
    line_count: int
    angle_degrees: float


def _align_image_iteration(image: np.ndarray, rho: float) -> AlignmentIterationResult:
    blurred = blur_image(image)
    edges = detect_edges(blurred)
    lines = extract_lines(edges, rho=rho)
    if not lines:
        return AlignmentIterationResult(image, 0, 0)
    draw_lines(blurred, lines=lines, title="lines_before_rotate")
    angle_degrees = _find_rotation_angle(lines)
    log.info(f"Rotation angle: {angle_degrees}")
    aligned_image = _rotate_by(image, angle_degrees)
    save_debug_image(aligned_image, title="aligned")
    return AlignmentIterationResult(aligned_image=aligned_image, line_count=len(lines), angle_degrees=angle_degrees)


def _rotate_by(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    # Get the image center and dimensions
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1)
    # Calculate the new bounding dimensions of the rotated image
    cos_angle = np.abs(rotation_matrix[0, 0])
    sin_angle = np.abs(rotation_matrix[0, 1])
    new_w = int(h * sin_angle + w * cos_angle)
    new_h = int(h * cos_angle + w * sin_angle)
    # Adjust the rotation matrix to take into account the translation
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    # Apply the warp affine with the new dimensions to preserve all pixels
    aligned_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
    return aligned_image


def _find_rotation_angle(lines: list[Line]) -> float:
    grid_lines = _get_grid_lines(lines)
    sum = count = 0
    if grid_lines.horizontal:
        horizontal_diff = np.mean([_horizontal_diff(line.theta) for line in grid_lines.horizontal])
        sum += horizontal_diff
        count += 1
    if grid_lines.vertical:
        vertical_diff = np.mean([_vertical_diff(line.theta) for line in grid_lines.vertical])
        sum += vertical_diff
        count += 1
    if count == 0:
        return 0
    angle = sum / count
    angle_degrees = np.degrees(angle)
    return angle_degrees


def _horizontal_diff(theta: float) -> float:
    return float(np.mod(theta + np.pi / 2, np.pi) - np.pi / 2)


def _vertical_diff(theta: float) -> float:
    return float(theta - np.pi / 2)


def _get_grid_lines(lines: list[Line]) -> GridLines:
    horizontal = []
    vertical = []
    for line in lines:
        if _is_horizontal_line(line):
            horizontal.append(line)
        else:
            vertical.append(line)
    return GridLines(horizontal=horizontal, vertical=vertical)


def _is_horizontal_line(line: Line) -> bool:
    if line.theta < np.pi / 4 or line.theta > 3 * np.pi / 4:
        return True
    return False


def blur_image(image: np.ndarray) -> np.ndarray:
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    save_debug_image(blurred, title="blurred")
    return blurred


def detect_edges(image: np.ndarray) -> np.ndarray:
    # Edge detection
    edges = cv2.Canny(image, 50, 150)
    save_debug_image(edges, title="edges")
    return edges


def extract_lines(edges: np.ndarray, rho: float = 1, theta: float = np.pi / 180, threshold: int = 100) -> list[Line]:
    # Find lines using Hough transform
    hough_lines = cv2.HoughLines(edges, rho=rho, theta=theta, threshold=threshold)
    if hough_lines is None:
        return []
    lines = []
    for line in hough_lines:
        rho, theta = line[0]
        line = Line(rho, theta)
        lines.append(line)
    return lines
