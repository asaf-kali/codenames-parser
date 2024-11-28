import logging
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np

from codenames_parser.common.utils.debug_util import (
    SEPARATOR,
    draw_lines,
    save_debug_image,
)
from codenames_parser.common.utils.models import GridLines, Line

log = logging.getLogger(__name__)

MIN_LINE_COUNT = 10
MIN_ROTATION_ANGLE = 0.01


@dataclass
class AlignmentResult:
    aligned_image: np.ndarray
    rotations: list[float]


def align_image(image: np.ndarray) -> AlignmentResult:
    # TODO: Handle tilt
    iter_count = 1
    rho = 1.0
    max_angle = 20.0
    rotations = []
    log.info(SEPARATOR)
    log.info("Starting image alignment...")
    while True:
        log.info(SEPARATOR)
        log.info(f"Align image iteration {iter_count}")
        result = _align_image_iteration(image, rho=rho, max_angle=max_angle)
        image = result.aligned_image
        rotations.append(result.rotation_degrees)
        if result.line_count < MIN_LINE_COUNT:
            break
        if abs(result.rotation_degrees) < MIN_ROTATION_ANGLE:
            break
        rho /= 1.2
        max_angle /= 4
        iter_count += 1
    log.info("Image alignment completed")
    return AlignmentResult(aligned_image=image, rotations=rotations)


class AlignmentIterationResult(NamedTuple):
    aligned_image: np.ndarray
    line_count: int
    rotation_degrees: float


def _align_image_iteration(image: np.ndarray, rho: float, max_angle: float) -> AlignmentIterationResult:
    blurred = blur_image(image)
    edges = detect_edges(blurred)
    lines = extract_lines(edges, rho=rho)
    if not lines:
        return AlignmentIterationResult(aligned_image=image, line_count=0, rotation_degrees=0)
    angle_degrees = _find_rotation_angle(lines, max_angle=max_angle)
    log.info(f"Rotation angle: {angle_degrees}")
    if abs(angle_degrees) < MIN_ROTATION_ANGLE:
        return AlignmentIterationResult(aligned_image=image, line_count=len(lines), rotation_degrees=0)
    draw_lines(blurred, lines=lines, title="lines before rotate")
    aligned_image = apply_rotation(image, angle_degrees)
    save_debug_image(aligned_image, title=f"aligned {angle_degrees:.2f} deg")
    return AlignmentIterationResult(aligned_image=aligned_image, line_count=len(lines), rotation_degrees=angle_degrees)


def apply_rotations(image: np.ndarray, rotations: list[float]) -> np.ndarray:
    for rotation in rotations:
        image = apply_rotation(image, rotation)
    return image


def apply_rotation(image: np.ndarray, angle_degrees: float) -> np.ndarray:
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


def _find_rotation_angle(lines: list[Line], max_angle: float) -> float:
    grid_lines = get_grid_lines(lines, max_angle=max_angle)
    angle_sum = 0.0
    count = 0
    if grid_lines.horizontal:
        horizontal_diff = np.mean([_horizontal_diff(line.theta) for line in grid_lines.horizontal])
        angle_sum += float(horizontal_diff)
        count += 1
    if grid_lines.vertical:
        vertical_diff = np.mean([_vertical_diff(line.theta) for line in grid_lines.vertical])
        angle_sum += float(vertical_diff)
        count += 1
    if count == 0:
        return 0
    angle_avg = angle_sum / count
    angle_degrees = np.degrees(angle_avg)
    return angle_degrees


def _horizontal_diff(theta: float) -> float:
    return float(theta - np.pi / 2)


def _vertical_diff(theta: float) -> float:
    return float(np.mod(theta + np.pi / 2, np.pi) - np.pi / 2)


def get_grid_lines(lines: list[Line], max_angle: float = 5) -> GridLines:
    horizontal = []
    vertical = []
    skipped = []
    for line in lines:
        if _is_horizontal_line(line, max_angle=max_angle):
            horizontal.append(line)
        elif _is_vertical_line(line, max_angle=max_angle):
            vertical.append(line)
        else:
            skipped.append(line)
            log.debug(f"Skipping non-grid line: {line}")
    log.info(f"Total lines: {len(lines)}")
    log.info(f"Max angle: {max_angle}°")
    log.info(f"Horizontal lines: {len(horizontal)}")
    log.info(f"Vertical lines: {len(vertical)}")
    log.info(f"Skipped lines: {len(skipped)}")
    return GridLines(horizontal=horizontal, vertical=vertical)


def _is_horizontal_line(line: Line, max_angle: float) -> bool:
    diff = _horizontal_diff(line.theta)
    return _is_grid_line(diff, max_angle=max_angle)


def _is_vertical_line(line: Line, max_angle: float) -> bool:
    diff = _vertical_diff(line.theta)
    return _is_grid_line(diff, max_angle=max_angle)


def _is_grid_line(diff: float, max_angle: float) -> bool:
    diff_degrees = np.degrees(diff)
    return abs(diff_degrees) < max_angle


def blur_image(image: np.ndarray, k_size: int = 5) -> np.ndarray:
    if len(image.shape) == 2:
        gray = image
    else:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    kernel = (k_size, k_size)
    blurred = cv2.GaussianBlur(gray, kernel, 0)
    save_debug_image(blurred, title=f"blurred k-{k_size}")
    return blurred


def detect_edges(image: np.ndarray, is_blurred: bool = True, threshold1: int = 50, threshold2: int = 150) -> np.ndarray:
    # Edge detection
    blurred = image if is_blurred else blur_image(image)
    edges = cv2.Canny(blurred, threshold1, threshold2)
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
