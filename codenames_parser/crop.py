import logging
from typing import NamedTuple

import numpy as np

from codenames_parser.align import (
    blur_image,
    detect_edges,
    extract_lines,
    get_grid_lines,
)
from codenames_parser.debugging.util import SEPARATOR, draw_lines, save_debug_image

log = logging.getLogger(__name__)


class Line(NamedTuple):
    rho: float  # distance from the origin
    theta: float  # angle in radians


class AxisBounds(NamedTuple):
    start: Line
    end: Line


def crop_image(image: np.ndarray) -> np.ndarray:
    """
    Crop the input image according to the main Hough lines.
    """
    log.info(SEPARATOR)
    log.info("Starting image cropping...")
    blurred = blur_image(image)
    edges = detect_edges(blurred)
    lines = extract_lines(edges, rho=1)
    grid_lines = get_grid_lines(lines, max_angle=1)
    horizontal_bounds = find_crop_bounds(lines=grid_lines.horizontal)
    vertical_bounds = find_crop_bounds(lines=grid_lines.vertical)
    x = [*horizontal_bounds, *vertical_bounds]
    draw_lines(image, lines=x, title="bounds")
    cropped = crop_by_bounds(image, horizontal_bounds=horizontal_bounds, vertical_bounds=vertical_bounds)
    return cropped


def crop_by_bounds(image: np.ndarray, horizontal_bounds: AxisBounds, vertical_bounds: AxisBounds) -> np.ndarray:
    """
    Crop the input image according to the given bounds.
    """
    start_x = _valid_rho(vertical_bounds.start.rho)
    end_x = _valid_rho(vertical_bounds.end.rho)
    start_y = _valid_rho(horizontal_bounds.start.rho)
    end_y = _valid_rho(horizontal_bounds.end.rho)
    cropped = image[start_x:end_x, start_y:end_y]
    save_debug_image(cropped, title="cropped")
    return cropped


def _valid_rho(rho: float) -> int:
    return max(0, int(rho))


def find_crop_bounds(lines: list[Line]) -> AxisBounds:
    """
    Find the crop bounds for the given axis.
    """
    # Sort lines by rho
    lines = sorted(lines, key=lambda x: x.rho)
    # Skip the first and last lines, which are possibly image borders.
    # We anyway expect the grid lines to be in the middle of the picture, so this should be fine.
    # TODO: Find a better way to handle this (check if the lines are close to the image borders)
    start = lines[1]
    end = lines[-2]
    return AxisBounds(start=start, end=end)