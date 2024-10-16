import logging

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from codenames_parser.color_map.mask import color_distance_mask
from codenames_parser.common.align import blur_image
from codenames_parser.common.debug_util import SEPARATOR, draw_boxes, save_debug_image
from codenames_parser.common.grid_detection import (
    GRID_HEIGHT,
    GRID_SIZE,
    GRID_WIDTH,
    crop_cells,
    deduplicate_boxes,
    filter_non_common_boxes,
    find_boxes,
)
from codenames_parser.common.models import Box, Color, Grid

log = logging.getLogger(__name__)

WHITE = Color(255, 255, 255)
CARD_RATIO = 1.8


def extract_cells(image: np.ndarray) -> Grid[np.ndarray]:
    log.info(SEPARATOR)
    log.info("Extracting card cells...")
    card_boxes = find_card_boxes(image)
    deduplicated_boxes = deduplicate_boxes(boxes=card_boxes)
    draw_boxes(image, boxes=deduplicated_boxes, title="boxes deduplicated")
    all_card_boxes = _complete_missing_boxes(deduplicated_boxes)
    draw_boxes(image, boxes=all_card_boxes, title=f"{GRID_SIZE} boxes")
    grid = crop_cells(image, boxes=all_card_boxes)
    return grid


def find_card_boxes(image: np.ndarray) -> list[Box]:
    blurred = blur_image(image)
    equalized = cv2.equalizeHist(blurred)
    save_debug_image(equalized, title="equalized")
    color_distance = color_distance_mask(image, color=WHITE, percentile=80)
    boxes = find_boxes(image=color_distance.filtered_negative, ratio_max=CARD_RATIO, min_size=10)
    draw_boxes(image, boxes=boxes, title="boxes raw")
    card_boxes = filter_non_common_boxes(boxes)
    draw_boxes(image, boxes=card_boxes, title="boxes filtered")
    return card_boxes


def _complete_missing_boxes(boxes: list[Box]) -> Grid[Box]:
    """
    Complete missing boxes in the list to reach the expected GRID_SIZE.
    Boxes might not be exactly aligned in a grid, so we can't assume constant row and column sizes.
    We need to understand which boxes are missing, and then try to assume their positions.
    """
    # Extract the centers of the boxes
    x_centers = np.array([box.x_center for box in boxes])
    y_centers = np.array([box.y_center for box in boxes])
    boxes_positions = np.column_stack((x_centers, y_centers))

    # Compute the expected grid positions
    min_x_center, max_x_center = np.min(x_centers), np.max(x_centers)
    min_y_center, max_y_center = np.min(y_centers), np.max(y_centers)

    expected_x_positions = np.linspace(min_x_center, max_x_center, GRID_WIDTH)
    expected_y_positions = np.linspace(min_y_center, max_y_center, GRID_HEIGHT)

    grid_positions = [(x, y) for y in expected_y_positions for x in expected_x_positions]

    # Build the cost matrix between detected boxes and grid positions
    num_boxes = len(boxes)
    num_grid_positions = GRID_SIZE
    cost_matrix = np.zeros((num_boxes, num_grid_positions))

    for i in range(num_boxes):
        for j in range(num_grid_positions):
            cost_matrix[i, j] = np.hypot(
                boxes_positions[i][0] - grid_positions[j][0],
                boxes_positions[i][1] - grid_positions[j][1],
            )

    # Use linear sum assignment to assign boxes to grid positions
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Map the assignments
    assigned_boxes = {col: boxes[row] for row, col in zip(row_ind, col_ind)}
    average_width = np.mean([box.x_max - box.x_min for box in boxes])
    average_height = np.mean([box.y_max - box.y_min for box in boxes])

    all_boxes = []
    for idx in range(num_grid_positions):
        x_center, y_center = grid_positions[idx]
        if idx in assigned_boxes:
            # Use the assigned box
            box = assigned_boxes[idx]
        else:
            # Create a new box at the expected position
            x_min = int(x_center - average_width / 2)
            x_max = int(x_center + average_width / 2)
            y_min = int(y_center - average_height / 2)
            y_max = int(y_center + average_height / 2)
            width = x_max - x_min
            height = y_max - y_min
            box = Box(x=x_min, y=y_min, w=width, h=height)
        all_boxes.append(box)
    grid = Grid.from_list(row_size=GRID_WIDTH, items=all_boxes)
    return grid
