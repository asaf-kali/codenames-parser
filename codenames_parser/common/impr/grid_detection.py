import logging
from typing import Iterable

import cv2
import numpy as np
from sklearn.cluster import KMeans

from codenames_parser.common.impr.crop import crop_by_box
from codenames_parser.common.utils.errors import NotEnoughBoxesError
from codenames_parser.common.utils.models import Box

log = logging.getLogger(__name__)
GRID_SIDE = 5
GRID_WIDTH = GRID_SIDE
GRID_HEIGHT = GRID_SIDE
GRID_SIZE = GRID_WIDTH * GRID_HEIGHT


def find_boxes(
    image: np.ndarray, expected_ratio: float = 1, max_ratio_diff: float = 0.2, min_size: int = 15
) -> list[Box]:
    ratio_max = expected_ratio + max_ratio_diff
    ratio_min = expected_ratio - max_ratio_diff
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


def deduplicate_boxes(boxes: list[Box], max_iou: float = 0.5) -> list[Box]:
    # Deduplicate boxes based on Intersection over Union (IoU)
    deduplicated_boxes: list[Box] = []
    for box in boxes:
        is_duplicate = False
        for existing_box in deduplicated_boxes:
            iou = _box_iou(box, existing_box)
            if iou > max_iou:
                is_duplicate = True
                break
        if not is_duplicate:
            deduplicated_boxes.append(box)
    return deduplicated_boxes


def crop_cells(image: np.ndarray, boxes: Iterable[Box]) -> list[np.ndarray]:
    return [crop_by_box(image, box=box) for box in boxes]


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
    log.info(f"Raw box count: {len(boxes)}")
    if len(boxes) < 5:
        raise NotEnoughBoxesError()
    common_area = _detect_common_box_area(boxes)
    filtered_boxes = [box for box in boxes if _is_common_box(box, common_area)]
    log.info(f"Filtered box count: {len(filtered_boxes)} (removed {len(boxes) - len(filtered_boxes)})")
    return filtered_boxes


def _detect_common_box_area(boxes: list[Box]) -> int:
    # Extract areas and reshape for clustering
    areas = np.array([box.area for box in boxes]).reshape(-1, 1)
    ratios = np.array([box.w / box.h for box in boxes]).reshape(-1, 1)
    vectors = np.concatenate([areas, ratios], axis=1)

    # Cluster the areas
    kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto")
    kmeans.fit(vectors)
    _log_kmeans_results(kmeans)

    # Get labels and cluster centers
    labels = kmeans.labels_
    area_cluster_centers = kmeans.cluster_centers_[:, 0]

    # Merge close clusters
    labels_merged = _merge_close_area_clusters(labels=labels, centers=area_cluster_centers)

    # Compute 40th and 60th percentiles of the areas
    percentiles = np.percentile(areas, q=[40, 60])
    log.info(f"Area 40th percentile: {percentiles[0]:.0f}")
    log.info(f"Area 60th percentile: {percentiles[1]:.0f}")

    # Filter clusters based on the percentiles
    areas_flatten = areas.flatten()
    filtered_labels = labels_merged[(areas_flatten >= percentiles[0]) & (areas_flatten <= percentiles[1])]

    # Find the cluster with the maximum number of boxes within the filtered range
    unique_labels, counts = np.unique(filtered_labels, return_counts=True)
    log.info(f"Filtered labels: {unique_labels}")
    common_cluster_label = unique_labels[np.argmax(counts)]
    log.info(f"Common cluster label: {common_cluster_label}")

    # Compute the mean area of the boxes in the common cluster
    common_areas = areas[labels_merged == common_cluster_label]
    common_area = int(np.mean(common_areas))
    log.info(f"Common area: {common_area}")
    return common_area


def _log_kmeans_results(kmeans: KMeans) -> None:
    log.info("KMeans results:")
    for label, center in enumerate(kmeans.cluster_centers_):
        count = np.sum(kmeans.labels_ == label)
        area, ratio = center[0], center[1]
        log.info(f"Cluster {label}: count={count:<3} area={area:<6.0f} ratio={ratio:.3f}")


def _is_common_box(box: Box, common_area: int, ratio_max: float = 1.5) -> bool:
    ratio_min = 1 / ratio_max
    if ratio_min > ratio_max:
        ratio_min, ratio_max = ratio_max, ratio_min
    ratio = box.area / common_area
    return ratio_min <= ratio <= ratio_max


def _merge_close_area_clusters(labels: np.ndarray, centers: Iterable[float], threshold: float = 0.07) -> np.ndarray:
    for i, center_i in enumerate(centers):
        for j, center_j in enumerate(centers):
            if i == j:
                continue
            diff = abs(center_i - center_j)
            if diff < threshold * center_i:
                log.info(f"Merging clusters {i} and {j} with diff={diff:.3f} (threshold={threshold})")
                labels[labels == j] = i
    return labels
