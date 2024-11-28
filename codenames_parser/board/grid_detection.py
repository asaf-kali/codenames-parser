import logging
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from codenames_parser.color_map.mask import color_distance_mask
from codenames_parser.common.impr.align import blur_image
from codenames_parser.common.impr.grid_detection import (
    GRID_HEIGHT,
    GRID_SIZE,
    GRID_WIDTH,
    deduplicate_boxes,
    filter_non_common_boxes,
    find_boxes,
)
from codenames_parser.common.utils.debug_util import (
    SEPARATOR,
    draw_boxes,
    save_debug_image,
)
from codenames_parser.common.utils.errors import (
    GridExtractionFailedError,
    NotEnoughBoxesError,
)
from codenames_parser.common.utils.models import Box, Color, Point

log = logging.getLogger(__name__)

WHITE = Color(255, 255, 255)
CARD_RATIO = 1.55
UNCERTAIN_BOX_FACTOR = 1.1
COLOR_MASK_PERCENTILES = [80, 75, 70, 65, 60, 50]


@dataclass
class RowColIndices:
    row: list[int]
    col: list[int]


def extract_boxes(image: np.ndarray) -> list[Box]:
    log.info(SEPARATOR)
    log.info("Extracting card cells...")
    for percentile in COLOR_MASK_PERCENTILES:
        log.info(f"Trying with percentile {percentile}")
        try:
            return _extract_cells_iteration(image, color_mask_percentile=percentile)
        except NotEnoughBoxesError:
            log.info(SEPARATOR)
            log.info("Not enough boxes to complete the grid, trying with a lower color mask percentile")
    log.error("Failed to extract card cells")
    raise GridExtractionFailedError()


def _extract_cells_iteration(image: np.ndarray, color_mask_percentile: int) -> list[Box]:
    card_boxes = find_card_boxes(image, percentile=color_mask_percentile)
    deduplicated_boxes = deduplicate_boxes(boxes=card_boxes)
    draw_boxes(image, boxes=deduplicated_boxes, title="boxes deduplicated")
    complete_card_boxes = _complete_missing_boxes(deduplicated_boxes)
    draw_boxes(image, boxes=complete_card_boxes, title=f"{GRID_SIZE} boxes")
    return complete_card_boxes


def find_card_boxes(image: np.ndarray, percentile: int) -> list[Box]:
    blurred = blur_image(image)
    equalized = cv2.equalizeHist(blurred)
    save_debug_image(equalized, title="equalized")
    color_distance = color_distance_mask(image, color=WHITE, percentile=percentile)
    boxes = find_boxes(image=color_distance.filtered_negative, expected_ratio=CARD_RATIO, max_ratio_diff=0.3)
    draw_boxes(image, boxes=boxes, title="boxes raw")
    card_boxes = filter_non_common_boxes(boxes)
    draw_boxes(image, boxes=card_boxes, title="boxes filtered")
    return card_boxes


def _complete_missing_boxes(boxes: list[Box]) -> list[Box]:
    """
    Complete missing boxes in the list to reach the expected GRID_SIZE.
    Boxes might not be exactly aligned in a grid, so we can't assume constant row and column sizes.
    We need to understand which boxes are missing, and then try to assume their positions,
    based on the found boxes in their row and column.
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

    # Check if we have enough boxes to complete the grid
    average_width = np.mean([box.x_max - box.x_min for box in boxes])
    average_height = np.mean([box.y_max - box.y_min for box in boxes])
    log.info(f"Average box width, height: ({average_width:.2f}, {average_height:.2f})")
    expected_x_diff = expected_x_positions[1] - expected_x_positions[0]
    expected_y_diff = expected_y_positions[1] - expected_y_positions[0]
    log.info(f"Expected x, y diff: ({expected_x_diff:.2f}, {expected_y_diff:.2f})")
    if expected_x_diff < average_width or expected_y_diff < average_height:
        raise NotEnoughBoxesError()

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
    assigned_boxes = {j: boxes[i] for i, j in zip(box_idx, grid_idx)}

    # Map the assignments
    grid_positions = _predict_missing_boxes_centers(assigned_boxes=assigned_boxes, grid_positions=grid_positions)
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
    return all_boxes


def _predict_missing_boxes_centers(assigned_boxes: dict[int, Box], grid_positions: list[Point]) -> list[Point]:
    """
    Predict the centers of missing boxes by looking at the average centers of its row and column.
    """
    missing_indices = set(range(GRID_SIZE)) - set(assigned_boxes.keys())
    log.info(f"Missing box indices: {missing_indices}")
    for i in missing_indices:
        row_col_indices = _get_row_col_indices(i)
        x_positions = [assigned_boxes[j].x_center for j in row_col_indices.col if j not in missing_indices]
        y_positions = [assigned_boxes[j].y_center for j in row_col_indices.row if j not in missing_indices]
        if not x_positions or not y_positions:
            log.info(f"Missing box {i} does not have enough neighbors")
            continue
        x_center = int(np.mean(x_positions))
        y_center = int(np.mean(y_positions))
        new_expected_center = Point(x_center, y_center)
        diff = new_expected_center - grid_positions[i]
        log.info(f"Predicted center for box {i}: {new_expected_center}, diff: {diff}")
        grid_positions[i] = new_expected_center
    return grid_positions


def _get_row_col_indices(box_index: int) -> RowColIndices:
    """
    Get the indices of all boxes in the same row and column as the given box index.
    """
    row = box_index // GRID_WIDTH
    col = box_index % GRID_WIDTH
    row_indices, col_indices = [], []
    for i in range(GRID_HEIGHT):
        col_indices.append(col + i * GRID_WIDTH)
    for i in range(GRID_WIDTH):
        row_indices.append(row * GRID_WIDTH + i)
    return RowColIndices(row=row_indices, col=col_indices)
