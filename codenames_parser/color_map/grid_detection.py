import logging

import numpy as np

from codenames_parser.color_map.consts import CODENAMES_COLORS
from codenames_parser.color_map.mask import color_distance_mask
from codenames_parser.common.debug_util import SEPARATOR, draw_boxes
from codenames_parser.common.grid_detection import (
    GRID_HEIGHT,
    GRID_SIZE,
    GRID_WIDTH,
    crop_cells,
    deduplicate_boxes,
    filter_non_common_boxes,
    find_boxes,
)
from codenames_parser.common.models import Box, Grid

log = logging.getLogger(__name__)


# pylint: disable=R0801
def extract_cells(image: np.ndarray) -> Grid[np.ndarray]:
    log.info(SEPARATOR)
    log.info("Extracting color cells...")
    card_boxes = find_color_boxes(image)
    deduplicated_boxes = deduplicate_boxes(boxes=card_boxes)
    draw_boxes(image, boxes=deduplicated_boxes, title="boxes deduplicated")
    all_card_boxes = _complete_missing_boxes(deduplicated_boxes)
    draw_boxes(image, boxes=all_card_boxes, title=f"{GRID_SIZE} boxes")
    grid = crop_cells(image, boxes=all_card_boxes)
    return grid


def find_color_boxes(image: np.ndarray) -> list[Box]:
    masks = [color_distance_mask(image, color=color) for color in CODENAMES_COLORS]
    boxes = []
    for mask in masks:
        color_boxes = find_boxes(image=mask.filtered_negative)
        draw_boxes(image, boxes=color_boxes, title="color boxes")
        boxes.extend(color_boxes)
    draw_boxes(image, boxes=boxes, title="boxes raw")
    color_boxes = filter_non_common_boxes(boxes)
    draw_boxes(image, boxes=color_boxes, title="boxes filtered")
    return color_boxes


def _complete_missing_boxes(boxes: list[Box]) -> Grid[Box]:
    num_rows, num_cols = GRID_HEIGHT, GRID_WIDTH

    # Collect x and y centers of existing boxes
    x_centers = [box.x + box.w / 2 for box in boxes]
    y_centers = [box.y + box.h / 2 for box in boxes]

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
    boxes_grid: Grid[Box] = Grid(row_size=num_cols)
    for row in range(num_rows):
        y_center = min_y_center + row * y_step
        box_row: list[Box] = []
        for col in range(num_cols):
            x_center = min_x_center + col * x_step
            x = int(x_center - avg_w / 2)
            y = int(y_center - avg_h / 2)
            box = Box(x, y, avg_w, avg_h)
            box_row.append(box)
        boxes_grid.append(box_row)

    return boxes_grid
