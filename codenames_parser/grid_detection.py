import logging

import cv2
import numpy as np

from codenames_parser.consts import CODENAMES_COLORS
from codenames_parser.crop import crop_by_box
from codenames_parser.debugging.util import SEPARATOR, draw_boxes
from codenames_parser.mask import color_distance_mask
from codenames_parser.models import Box, Grid

log = logging.getLogger(__name__)

GRID_SIZE = 5


def extract_cells(image: np.ndarray) -> Grid[np.ndarray]:
    log.info(SEPARATOR)
    log.info("Extracting cells...")
    card_boxes = find_card_boxes(image)
    deduplicated_boxes = _deduplicate_boxes(boxes=card_boxes)
    draw_boxes(image, boxes=deduplicated_boxes, title="boxes deduplicated")
    all_card_boxes = _complete_missing_boxes(deduplicated_boxes)
    draw_boxes(image, boxes=all_card_boxes, title="25 boxes")
    grid = _crop_cells(image, all_card_boxes)
    return grid


def _crop_cells(image: np.ndarray, all_card_boxes: list[Box]) -> Grid[np.ndarray]:
    grid = Grid(row_size=GRID_SIZE)
    for i in range(GRID_SIZE):
        row = []
        for j in range(GRID_SIZE):
            box = all_card_boxes[i * 5 + j]
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
    deduplicated_boxes = []
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
    if len(boxes) == 25:
        return boxes

    # Get x and y centers of existing boxes
    x_centers = np.array([box.x + box.w / 2 for box in boxes])
    y_centers = np.array([box.y + box.h / 2 for box in boxes])

    # Cluster x and y centers into 5 clusters each
    from sklearn.cluster import KMeans

    kmeans_x = KMeans(n_clusters=5, random_state=0)
    kmeans_y = KMeans(n_clusters=5, random_state=0)
    x_labels = kmeans_x.fit_predict(x_centers.reshape(-1, 1))  # noqa: F841
    y_labels = kmeans_y.fit_predict(y_centers.reshape(-1, 1))  # noqa: F841

    # Get sorted cluster centers
    x_cluster_centers = sorted(kmeans_x.cluster_centers_.flatten())
    y_cluster_centers = sorted(kmeans_y.cluster_centers_.flatten())

    # Compute average width and height of the boxes
    avg_w = int(np.mean([box.w for box in boxes]))
    avg_h = int(np.mean([box.h for box in boxes]))

    # Generate the grid of boxes based on cluster centers
    grid_boxes = []
    for y_center in y_cluster_centers:
        for x_center in x_cluster_centers:
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
