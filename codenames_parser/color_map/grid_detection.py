import logging

import cv2
import numpy as np

from codenames_parser.color_map.consts import CODENAMES_COLORS
from codenames_parser.color_map.crop import crop_by_box
from codenames_parser.color_map.mask import color_distance_mask
from codenames_parser.color_map.models import Box, Grid
from codenames_parser.common.debug_util import SEPARATOR, draw_boxes

log = logging.getLogger(__name__)

GRID_SIDE = 5
GRID_SIZE = GRID_SIDE * GRID_SIDE


def extract_cells(image: np.ndarray) -> Grid[np.ndarray]:
    log.info(SEPARATOR)
    log.info("Extracting cells...")
    card_boxes = find_card_boxes(image)
    deduplicated_boxes = _deduplicate_boxes(boxes=card_boxes)
    draw_boxes(image, boxes=deduplicated_boxes, title="boxes deduplicated")
    all_card_boxes = _complete_missing_boxes(deduplicated_boxes)
    draw_boxes(image, boxes=all_card_boxes, title=f"{GRID_SIZE} boxes")
    grid = _crop_cells(image, all_card_boxes)
    return grid


def _crop_cells(image: np.ndarray, all_card_boxes: list[Box]) -> Grid[np.ndarray]:
    grid: Grid[np.ndarray] = Grid(row_size=GRID_SIDE)
    for i in range(GRID_SIDE):
        row = []
        for j in range(GRID_SIDE):
            box = all_card_boxes[i * GRID_SIDE + j]
            cell = crop_by_box(image, box)
            row.append(cell)
        grid.append(row)
    return grid


def find_card_boxes(image: np.ndarray) -> list[Box]:
    masks = [color_distance_mask(image, color=color) for color in CODENAMES_COLORS]
    boxes = []
    for mask in masks:
        color_boxes = find_boxes(image=mask.filtered_negative)
        boxes.extend(color_boxes)
    draw_boxes(image, boxes=boxes, title="boxes raw")
    card_boxes = _filter_card_boxes(boxes)
    draw_boxes(image, boxes=card_boxes, title="boxes filtered")
    return card_boxes


def _deduplicate_boxes(boxes: list[Box]) -> list[Box]:
    # Deduplicate boxes based on Intersection over Union (IoU)
    deduplicated_boxes: list[Box] = []
    for box in boxes:
        is_duplicate = False
        for existing_box in deduplicated_boxes:
            iou = _box_iou(box, existing_box)
            if iou > 0.5:
                is_duplicate = True
                break
        if not is_duplicate:
            deduplicated_boxes.append(box)
    return deduplicated_boxes


def _box_iou(box1: Box, box2: Box) -> float:
    # Compute the Intersection over Union (IoU) of two boxes
    x_left = max(box1.x, box2.x)
    y_top = max(box1.y, box2.y)
    x_right = min(box1.x + box1.w, box2.x + box2.w)
    y_bottom = min(box1.y + box1.h, box2.y + box2.h)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = box1.area + box2.area - intersection_area
    iou = intersection_area / union_area
    return iou


def _complete_missing_boxes(boxes: list[Box]) -> list[Box]:
    if len(boxes) == GRID_SIZE:
        return boxes

    num_rows, num_cols = GRID_SIDE, GRID_SIDE

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
    grid_boxes = []
    for row in range(num_rows):
        y_center = min_y_center + row * y_step
        for col in range(num_cols):
            x_center = min_x_center + col * x_step
            x = int(x_center - avg_w / 2)
            y = int(y_center - avg_h / 2)
            grid_boxes.append(Box(x, y, avg_w, avg_h))

    return grid_boxes


def _filter_card_boxes(boxes: list[Box]) -> list[Box]:
    common_area = _detect_common_box_area(boxes)
    filtered_boxes = [box for box in boxes if _is_card_box(box, common_area)]
    return filtered_boxes


def _is_card_box(box: Box, common_area: int, ratio_diff: float = 0.2) -> bool:
    ratio_min, ratio_max = 1 - ratio_diff, 1 + ratio_diff
    ratio = box.area / common_area
    return ratio_min <= ratio <= ratio_max


def _detect_common_box_area(boxes: list[Box]) -> int:
    areas = [box.area for box in boxes]
    percentile_50 = np.percentile(areas, q=50)
    return int(percentile_50)


def find_boxes(image: np.ndarray, ratio_diff: float = 0.2, min_size: int = 10) -> list[Box]:
    ratio_min, ratio_max = 1 - ratio_diff, 1 + ratio_diff
    # Convert the mask to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find contours in the grayscale mask
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        # Get the bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out non-square-like contours by aspect ratio and minimum size
        aspect_ratio = w / float(h)
        if ratio_min <= aspect_ratio <= ratio_max and w > min_size and h > min_size:
            box = Box(x, y, w, h)
            bounding_boxes.append(box)
    return bounding_boxes
