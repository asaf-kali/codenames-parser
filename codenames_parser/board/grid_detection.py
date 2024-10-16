import logging

import cv2
import numpy as np

from codenames_parser.color_map.mask import color_distance_mask
from codenames_parser.common.align import blur_image
from codenames_parser.common.debug_util import SEPARATOR, draw_boxes, save_debug_image
from codenames_parser.common.grid_detection import (
    GRID_SIZE,
    crop_cells,
    deduplicate_boxes,
    filter_non_common_boxes,
    find_boxes,
)
from codenames_parser.common.models import Box, Color, Grid

log = logging.getLogger(__name__)

WHITE = Color(255, 255, 255)


def extract_cells(image: np.ndarray) -> Grid[np.ndarray]:
    log.info(SEPARATOR)
    log.info("Extracting card cells...")
    card_boxes = find_card_boxes(image)
    deduplicated_boxes = deduplicate_boxes(boxes=card_boxes)
    draw_boxes(image, boxes=deduplicated_boxes, title="boxes deduplicated")
    all_card_boxes = _complete_missing_boxes(deduplicated_boxes)
    draw_boxes(image, boxes=all_card_boxes, title=f"{GRID_SIZE} boxes")
    grid = crop_cells(image, all_card_boxes)
    return grid


def find_card_boxes(image: np.ndarray) -> list[Box]:
    blurred = blur_image(image)
    equalized = cv2.equalizeHist(blurred)
    save_debug_image(equalized, title="equalized")
    color_distance = color_distance_mask(image, color=WHITE, percentile=80)
    boxes = find_boxes(image=color_distance.filtered_negative)
    draw_boxes(image, boxes=boxes, title="boxes raw")
    card_boxes = filter_non_common_boxes(boxes)
    draw_boxes(image, boxes=card_boxes, title="boxes filtered")
    return card_boxes


def _complete_missing_boxes(boxes: list[Box]) -> list[Box]:
    """
    Complete missing boxes in the list to reach the expected GRID_SIZE.
    Boxes might not be exactly aligned in a grid, so we can't assume constant row and column sizes.
    We need to understand which boxes are missing, and then try to assume their positions.
    """
    return boxes
