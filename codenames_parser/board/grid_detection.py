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
CARD_RATIO_MAX = 1.8
UNCERTAIN_BOX_FACTOR = 1.3


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
    boxes = find_boxes(image=color_distance.filtered_negative, ratio_max=CARD_RATIO_MAX, min_size=10)
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
            diff_x = boxes_positions[i][0] - grid_positions[j][0]
            diff_y = boxes_positions[i][1] - grid_positions[j][1]
            cost_matrix[i, j] = np.hypot(diff_x, diff_y)

    # Use linear sum assignment to assign boxes to grid positions
    box_idx, grid_idx = linear_sum_assignment(cost_matrix)

    # Map the assignments
    assigned_boxes = {col: boxes[row] for row, col in zip(box_idx, grid_idx)}
    missing_indexes = set(range(num_grid_positions)) - set(grid_idx)
    log.info(f"Missing box indexes: {missing_indexes}")
    average_width = np.mean([box.x_max - box.x_min for box in boxes])
    average_height = np.mean([box.y_max - box.y_min for box in boxes])
    log.info(f"Average box width, height: ({average_width:.2f}, {average_height:.2f})")
    width_uncertain = average_width * UNCERTAIN_BOX_FACTOR
    height_uncertain = average_height * UNCERTAIN_BOX_FACTOR
    width_offset = (width_uncertain - average_width) / 2
    height_offset = (height_uncertain - average_height) / 2

    all_boxes = []
    for idx in range(num_grid_positions):
        x_center, y_center = grid_positions[idx]
        if idx in assigned_boxes:
            # Use the assigned box
            box = assigned_boxes[idx]
        else:
            # Create a new box at the expected position
            x_min = x_center - average_width / 2
            y_min = y_center - average_height / 2
            # Re-center the box
            x_min -= width_offset
            y_min -= height_offset
            box = Box(x=int(x_min), y=int(y_min), w=int(width_uncertain), h=int(height_uncertain))
        all_boxes.append(box)
    grid = Grid.from_list(row_size=GRID_WIDTH, items=all_boxes)
    return grid
