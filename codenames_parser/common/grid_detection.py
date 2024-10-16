import cv2
import numpy as np

from codenames_parser.common.crop import crop_by_box
from codenames_parser.common.models import Box, Grid

GRID_SIDE = 5
GRID_WIDTH = GRID_SIDE
GRID_HEIGHT = GRID_SIDE
GRID_SIZE = GRID_WIDTH * GRID_HEIGHT


def find_boxes(image: np.ndarray, ratio_max: float = 1.2, min_size: int = 10) -> list[Box]:
    ratio_min = 1 / ratio_max
    if ratio_min > ratio_max:
        ratio_min, ratio_max = ratio_max, ratio_min
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


def deduplicate_boxes(boxes: list[Box]) -> list[Box]:
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


def crop_cells(image: np.ndarray, boxes: Grid[Box]) -> Grid[np.ndarray]:
    cells = [crop_by_box(image, box=box) for box in boxes]
    grid = Grid.from_list(row_size=GRID_WIDTH, items=cells)
    return grid


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


def filter_non_common_boxes(boxes: list[Box]) -> list[Box]:
    common_area = _detect_common_box_area(boxes)
    filtered_boxes = [box for box in boxes if _is_common_box(box, common_area)]
    return filtered_boxes


def _detect_common_box_area(boxes: list[Box]) -> int:
    # TODO: This is a naive implementation, need to improve (e.g. use clustering to find common box)
    areas = [box.area for box in boxes]
    percentile_50 = np.percentile(areas, q=50)
    return int(percentile_50)


def _is_common_box(box: Box, common_area: int, ratio_diff: float = 0.2) -> bool:
    ratio_min, ratio_max = 1 - ratio_diff, 1 + ratio_diff
    ratio = box.area / common_area
    return ratio_min <= ratio <= ratio_max
