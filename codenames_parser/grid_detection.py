import logging

import cv2
import numpy as np

from codenames_parser.consts import CODENAMES_COLORS
from codenames_parser.crop import crop_by_box
from codenames_parser.debugging.util import SEPARATOR, draw_boxes
from codenames_parser.mask import color_distance_mask
from codenames_parser.models import Box

log = logging.getLogger(__name__)


def extract_cells(image: np.ndarray) -> list[np.ndarray]:
    log.info(SEPARATOR)
    log.info("Extracting cells...")
    card_boxes = find_card_boxes(image)
    deduplicated_boxes = _deduplicate_boxes(boxes=card_boxes)
    draw_boxes(image, boxes=deduplicated_boxes, title="boxes deduplicated")
    all_card_boxes = _complete_missing_boxes(deduplicated_boxes)
    draw_boxes(image, boxes=all_card_boxes, title="25 boxes")
    cropped_cells = [crop_by_box(image, box=box) for box in all_card_boxes]
    return cropped_cells


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
    # TODO: Implement deduplication logic
    # avg_area = float(np.mean([box.area for box in boxes]))
    pass


def _are_boxes_duplicates(box1: Box, box2: Box, avg_area: float) -> bool:
    # TODO: Implement deduplication logic
    return False


def _complete_missing_boxes(boxes: list[Box]) -> list[Box]:
    if len(boxes) == 25:
        return boxes


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
