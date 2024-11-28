import logging
from dataclasses import dataclass

import numpy as np
from codenames.duet.card import DuetColor
from codenames.generic.card import CardColor

from codenames_parser.color_map.color_translator import get_board_colors
from codenames_parser.color_map.mask import color_distance_mask
from codenames_parser.common.impr.align import detect_edges
from codenames_parser.common.impr.grid_detection import (
    GRID_HEIGHT,
    GRID_SIZE,
    GRID_WIDTH,
    crop_cells,
    deduplicate_boxes,
    filter_non_common_boxes,
    find_boxes,
)
from codenames_parser.common.utils.debug_util import SEPARATOR, draw_boxes
from codenames_parser.common.utils.errors import NotEnoughBoxesError
from codenames_parser.common.utils.models import Box, Point

log = logging.getLogger(__name__)

CenterDistances = dict[Box, float]


@dataclass
class GridFit:
    grid: list[Box]
    area_fit: float


# pylint: disable=R0801
def extract_cells(image: np.ndarray, color_type: type[CardColor]) -> list[np.ndarray]:
    log.info(SEPARATOR)
    log.info("Extracting color cells...")
    card_boxes = _find_color_boxes(image, color_type=color_type)
    # Filter outliers
    center_distances = _mean_center_distances(boxes=card_boxes)
    no_outliers = _filter_outliers(distances=center_distances)
    draw_boxes(image, boxes=no_outliers, title="boxes no outliers")
    # Deduplicate boxes
    deduplicated_boxes = deduplicate_boxes(boxes=no_outliers)
    draw_boxes(image, boxes=deduplicated_boxes, title="boxes deduplicated")
    # Complete missing boxes
    all_color_boxes = _detect_all_boxes(image, boxes=deduplicated_boxes, distances=center_distances)
    # Crop cells
    grid = crop_cells(image, boxes=all_color_boxes)
    return grid


def _find_color_boxes(image: np.ndarray, color_type: type[CardColor]) -> list[Box]:
    board_colors = get_board_colors(color_type=color_type)
    masks = [color_distance_mask(image, color=color) for color in board_colors]
    boxes = []
    expected_ratio = _get_expected_color_box_ratio(color_type=color_type)
    for mask in masks:
        edges = detect_edges(mask.filtered_negative)
        color_boxes = find_boxes(image=edges, expected_ratio=expected_ratio)
        draw_boxes(image, boxes=color_boxes, title="color boxes")
        boxes.extend(color_boxes)
    draw_boxes(image, boxes=boxes, title="boxes raw")
    color_boxes = filter_non_common_boxes(boxes)
    draw_boxes(image, boxes=color_boxes, title="boxes size filtered")
    return color_boxes


def _mean_center_distances(boxes: list[Box]) -> CenterDistances:
    # Compute the average center
    x_centers = [box.x_center for box in boxes]
    y_centers = [box.y_center for box in boxes]
    avg_center = Point(x=int(np.mean(x_centers)), y=int(np.mean(y_centers)))
    # Measure distance from each box to the average center
    distances = {}
    for box in boxes:
        distance = box.center.distance(other=avg_center)
        distances[box] = distance
    distances = dict(sorted(distances.items(), key=lambda x: -x[1]))
    return distances


def _filter_outliers(distances: CenterDistances) -> list[Box]:
    # Compute the average distance
    avg_distance = np.mean(list(distances.values()))
    # Compute the standard deviation
    std_distance = np.std(list(distances.values()))
    # Filter out boxes that are too far away
    outliers = {box for box, distance in distances.items() if distance >= avg_distance + 2 * std_distance}
    no_outliers = [box for box in distances.keys() if box not in outliers]
    for outlier in outliers:
        distances.pop(outlier)
    return no_outliers


def _detect_all_boxes(image: np.ndarray, boxes: list[Box], distances: CenterDistances) -> list[Box]:
    """
    Given deduplicated boxes and their distances from the average center, in iteration:
        1. all_boxes = Complete missing boxes
            1.a If there are intersections between boxes in all_boxes, this is not a valid grid, break the loop.
        2. Calculate area fit (the intersection between boxes and all_boxes)
            2.a If the area fit is too low, remove the box with the highest distance and go back to step 1.
            2.b If the area fit is good, return.
    """
    if len(boxes) < 5:
        raise NotEnoughBoxesError("Not enough boxes.")
    # Sort boxes by distance from the average center (highest first)
    boxes = sorted(boxes, key=lambda x: -distances[x])
    grid_fits = []
    for i in range(len(boxes) - 4):
        all_boxes = _complete_missing_boxes(boxes=boxes[i:])
        draw_boxes(image, boxes=all_boxes, title=f"all boxes iter {i + 1}")
        if not _is_legal_grid(boxes=all_boxes):
            break
        area_fit = _calculate_area_fit(boxes=boxes, all_boxes=all_boxes)
        grid_fit = GridFit(grid=all_boxes, area_fit=area_fit)
        grid_fits.append(grid_fit)
    if not grid_fits:
        raise NotEnoughBoxesError("No valid grid was found.")
    # Log results
    fit_values = [round(fit.area_fit * 100, 3) for fit in grid_fits]
    log.info(f"Fit values: {fit_values}")
    # Pick the best fit
    best_fit = max(grid_fits, key=lambda x: x.area_fit)
    if best_fit.area_fit > grid_fits[0].area_fit:
        log.info("Area fit was useful!")
    all_card_boxes = best_fit.grid
    draw_boxes(image, boxes=all_card_boxes, title=f"{GRID_SIZE} boxes")
    return all_card_boxes


def _calculate_area_fit(boxes: list[Box], all_boxes: list[Box]) -> float:
    """
    the total intersection between boxes and all_boxes
    """
    total_area = 0
    boxes_total = sum(box.area for box in boxes)
    for box_a in boxes:
        for box_b in all_boxes:
            if not box_a.overlaps(other=box_b):
                continue
            intersection = box_a.intersection(other=box_b)
            total_area += intersection.area
    fit_ratio = total_area / boxes_total
    return fit_ratio


def _is_legal_grid(boxes: list[Box]) -> bool:
    """
    Check if the grid is legal by checking if there are intersections between boxes.
    Assume boxes are exactly the same size, and in constant distances.
    """
    for i, box1 in enumerate(boxes):
        for j, box2 in enumerate(boxes):
            if i == j:
                continue
            if box1.overlaps(other=box2):
                return False
    return True


def _complete_missing_boxes(boxes: list[Box]) -> list[Box]:
    num_rows, num_cols = GRID_HEIGHT, GRID_WIDTH

    # Collect x and y centers of existing boxes
    x_centers = [box.x_center for box in boxes]
    y_centers = [box.y_center for box in boxes]

    # Compute average width and height of the boxes
    avg_w = int(np.mean([box.w for box in boxes]))
    avg_h = int(np.mean([box.h for box in boxes]))

    # Find min and max x and y centers
    min_x_center, max_x_center = min(x_centers), max(x_centers)
    min_y_center, max_y_center = min(y_centers), max(y_centers)

    # Compute the step sizes for x and y to create the grid
    x_step = (max_x_center - min_x_center) / (num_cols - 1)
    y_step = (max_y_center - min_y_center) / (num_rows - 1)

    # Generate the grid of boxes based on min/max centers and step sizes
    all_boxes = []
    for row in range(num_rows):
        y_center = min_y_center + row * y_step
        for col in range(num_cols):
            x_center = min_x_center + col * x_step
            x = int(x_center - avg_w / 2)
            y = int(y_center - avg_h / 2)
            box = Box(x, y, avg_w, avg_h)
            all_boxes.append(box)
    return all_boxes


def _get_expected_color_box_ratio(color_type: type[CardColor]) -> float:
    if color_type == DuetColor:
        return 1.4
    return 1
