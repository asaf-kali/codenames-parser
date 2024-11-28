import logging
from typing import NamedTuple

import cv2
import numpy as np

from codenames_parser.common.impr.align import (
    detect_edges,
    extract_lines,
    get_grid_lines,
)
from codenames_parser.common.utils.debug_util import (
    SEPARATOR,
    draw_lines,
    draw_polyline,
    save_debug_image,
)
from codenames_parser.common.utils.models import Box, Line, Point

log = logging.getLogger(__name__)


class AxisBounds(NamedTuple):
    start: Line
    end: Line


def crop_image(image: np.ndarray, min_crop_ratio: float = 0.4) -> np.ndarray:
    """
    Crop the input image according to the main Hough lines.

    Args:
        image: The input image.
        min_crop_ratio: The minimum ratio of the cropped image to the original image.
        Meaning, if the cropped image is smaller than the original image by this ratio, the cropping is skipped.
    """
    log.info(SEPARATOR)
    log.info("Starting image cropping...")
    edges = detect_edges(image, is_blurred=False)
    lines = extract_lines(edges, rho=1)
    grid_lines = get_grid_lines(lines, max_angle=1)
    draw_lines(image, lines=grid_lines, title="crop grid lines")
    try:
        horizontal_bounds = find_crop_bounds(lines=grid_lines.horizontal)
        vertical_bounds = find_crop_bounds(lines=grid_lines.vertical)
    except IndexError:
        log.info("Missing grid lines, skipping cropping")
        return image
    x = [*horizontal_bounds, *vertical_bounds]
    draw_lines(image, lines=x, title="crop bounds")
    cropped = crop_by_bounds(
        image, horizontal_bounds=horizontal_bounds, vertical_bounds=vertical_bounds, min_crop_ratio=min_crop_ratio
    )
    return cropped


def crop_by_bounds(
    image: np.ndarray, horizontal_bounds: AxisBounds, vertical_bounds: AxisBounds, min_crop_ratio: float
) -> np.ndarray:
    """
    Crop the input image according to the given bounds.
    """
    start_x = _valid_rho(vertical_bounds.start.rho)
    end_x = _valid_rho(vertical_bounds.end.rho)
    start_y = _valid_rho(horizontal_bounds.start.rho)
    end_y = _valid_rho(horizontal_bounds.end.rho)
    width_cropped, height_cropped = end_x - start_x, end_y - start_y
    width_original, height_original = image.shape[1], image.shape[0]
    width_ratio = width_cropped / width_original
    height_ratio = height_cropped / height_original
    log.info(f"Original image size: {width_original}x{height_original}")
    log.info(f"Cropped image size: {width_cropped}x{height_cropped}")
    log.info(f"Cropping ratio: {width_ratio:.2f}x{height_ratio:.2f}")
    if width_ratio < min_crop_ratio or height_ratio < min_crop_ratio:
        log.info("Cropping ratio is too low, skipping cropping")
        return image
    cropped = image[start_y:end_y, start_x:end_x]
    save_debug_image(cropped, title="cropped")
    return cropped


def crop_by_box(image: np.ndarray, box: Box) -> np.ndarray:
    """
    Crop the input image according to the given box.
    """
    if box.x < 0:
        box.w += box.x
        box.x = 0
    if box.y < 0:
        box.h += box.y
        box.y = 0
    cropped = image[box.y : box.y + box.h, box.x : box.x + box.w]
    # save_debug_image(cropped, title="cropped cell")
    return cropped


def _valid_rho(rho: float) -> int:
    return max(0, int(rho))


def find_crop_bounds(lines: list[Line]) -> AxisBounds:
    """
    Find the crop bounds for the given axis.
    """
    # Sort lines by rho
    lines = sorted(lines, key=lambda x: x.rho)
    start = lines[0]
    end = lines[-1]
    return AxisBounds(start=start, end=end)


def rotated_crop(image: np.ndarray, angle: float, top_left: Point, size: tuple[int, int]) -> np.ndarray:
    """
    Crop a region from the image, taking rotation into account.

    Args:
        image (np.ndarray): Original source image.
        angle (float): Rotation angle of the region.
        top_left (Point): Top-left corner location of the region.
        size (tuple[int, int]): Size of the region (height, width).

    Returns:
        np.ndarray: Cropped and straightened matched region from the source image.
    """
    # Get the size of the rotated template
    height, width = size
    vrt_center, hrz_center = height / 2, width / 2
    top_left_x, top_left_y = top_left
    # Define the corners of the template relative to its center
    corners = np.array(
        [
            [-hrz_center, -vrt_center],
            [hrz_center, -vrt_center],
            [hrz_center, vrt_center],
            [-hrz_center, vrt_center],
        ]
    )
    # Rotation matrix
    angle_rad = np.deg2rad(-angle)
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    rotated_corners = np.dot(corners, rotation_matrix.T)
    matched_corners = rotated_corners + np.array([top_left_x + hrz_center, top_left_y + vrt_center])
    dst_points = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    # Source points are the matched corners
    src_points = matched_corners.astype(np.float32)
    draw_polyline(image, points=src_points, title="matched region", important=True)

    # Compute the perspective transform matrix
    perspective_t = cv2.getPerspectiveTransform(src_points, dst_points)
    # Apply the perspective transform to get the straightened image
    cropped_image = cv2.warpPerspective(image, M=perspective_t, dsize=(width, height))
    return cropped_image
