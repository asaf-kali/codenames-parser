import logging
from typing import NamedTuple

import numpy as np

from codenames_parser.common.align import (
    blur_image,
    detect_edges,
    extract_lines,
    get_grid_lines,
)
from codenames_parser.common.debug_util import SEPARATOR, draw_lines, save_debug_image
from codenames_parser.common.models import Box, Line

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
    blurred = blur_image(image)
    edges = detect_edges(blurred)
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
