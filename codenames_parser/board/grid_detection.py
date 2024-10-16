import logging
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from codenames_parser.color_map.mask import color_distance_mask
from codenames_parser.common.align import blur_image
from codenames_parser.common.debug_util import SEPARATOR, draw_boxes, save_debug_image
from codenames_parser.common.errors import GridExtractionFailedError, MissingGridError
from codenames_parser.common.grid_detection import (
    GRID_HEIGHT,
    GRID_SIZE,
    GRID_WIDTH,
    crop_cells,
    deduplicate_boxes,
    filter_non_common_boxes,
    find_boxes,
)
from codenames_parser.common.models import Box, Color, Grid, Point

log = logging.getLogger(__name__)

WHITE = Color(255, 255, 255)
CARD_RATIO_MAX = 1.8
UNCERTAIN_BOX_FACTOR = 1.2
COLOR_MASK_PERCENTILES = [80, 70, 60, 50, 40]


@dataclass
class RowColIndexes:
    row: list[int]
    col: list[int]


def extract_cells(image: np.ndarray) -> Grid[np.ndarray]:
    log.info(SEPARATOR)
    log.info("Extracting card cells...")
    for percentile in COLOR_MASK_PERCENTILES:
        log.info(f"Trying with percentile {percentile}")
        try:
            return _extract_cells_iteration(image, color_mask_percentile=percentile)
        except MissingGridError:
            pass
    log.error("Failed to extract card cells")
    raise GridExtractionFailedError()


def _extract_cells_iteration(image: np.ndarray, color_mask_percentile: int) -> Grid[np.ndarray]:
    card_boxes = find_card_boxes(image, percentile=color_mask_percentile)
    deduplicated_boxes = deduplicate_boxes(boxes=card_boxes)
    draw_boxes(image, boxes=deduplicated_boxes, title="boxes deduplicated")
    all_card_boxes = _complete_missing_boxes(deduplicated_boxes)
    draw_boxes(image, boxes=all_card_boxes, title=f"{GRID_SIZE} boxes")
    grid = crop_cells(image, boxes=all_card_boxes)
    return grid


def find_card_boxes(image: np.ndarray, percentile: int) -> list[Box]:
    blurred = blur_image(image)
    equalized = cv2.equalizeHist(blurred)
    save_debug_image(equalized, title="equalized")
    color_distance = color_distance_mask(image, color=WHITE, percentile=percentile)
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

    grid_positions = [Point(x, y) for y in expected_y_positions for x in expected_x_positions]

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
    assigned_boxes = {j: boxes[i] for i, j in zip(box_idx, grid_idx)}
    grid_positions = _predict_missing_boxes_centers(assigned_boxes=assigned_boxes, grid_positions=grid_positions)
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


def _predict_missing_boxes_centers(assigned_boxes: dict[int, Box], grid_positions: list[Point]) -> list[Point]:
    """
    Predict the centers of missing boxes by looking at the average centers of its row and column.
    """
    missing_indexes = set(range(GRID_SIZE)) - set(assigned_boxes.keys())
    log.info(f"Missing box indexes: {missing_indexes}")
    for i in missing_indexes:
        row_col_indexes = _get_row_col_indexes(i)
        x_positions = [assigned_boxes[j].x_center for j in row_col_indexes.col if j not in missing_indexes]
        y_positions = [assigned_boxes[j].y_center for j in row_col_indexes.row if j not in missing_indexes]
        if not x_positions or not y_positions:
            log.error(f"Missing box {i} has no neighbors")
            raise MissingGridError()
        x_center = int(np.mean(x_positions))
        y_center = int(np.mean(y_positions))
        new_expected_center = Point(x_center, y_center)
        diff = new_expected_center - grid_positions[i]
        log.info(f"Predicted center for box {i}: {new_expected_center}, diff: {diff}")
        grid_positions[i] = new_expected_center
    return grid_positions


def _get_row_col_indexes(box_index: int) -> RowColIndexes:
    """
    Get the indexes of all boxes in the same row and column as the given box index.
    """
    row = box_index // GRID_WIDTH
    col = box_index % GRID_WIDTH
    row_indexes, col_indexes = [], []
    for i in range(GRID_HEIGHT):
        col_indexes.append(col + i * GRID_WIDTH)
    for i in range(GRID_WIDTH):
        row_indexes.append(row * GRID_WIDTH + i)
    return RowColIndexes(row=row_indexes, col=col_indexes)
