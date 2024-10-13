import logging

import cv2
import numpy as np

from codenames_parser.align import align_image, blur_image, detect_edges, extract_lines
from codenames_parser.debugging.util import draw_lines, save_debug_image
from codenames_parser.models import Line

log = logging.getLogger(__name__)


def extract_cells(image: np.ndarray) -> list[list[np.ndarray]]:
    """
    Extracts the individual cells from the perspective-corrected image.
    Steps:
    1. Detect edges using Canny edge detection.
    2. Align the image such that the grid lines are horizontal and vertical.
    3. Find the intersections of the grid lines.
    4. Extract the cell images.

    Args:
        image (np.ndarray): The perspective-corrected image.

    Returns:
        list[np.ndarray]: A list of cell images.
    """
    aligned_image = align_image(image)
    blurred = blur_image(aligned_image)
    edges = detect_edges(blurred)
    lines = extract_lines(edges, rho=0.2)
    draw_lines(aligned_image, lines, title="lines_after_alignment")
    # lines_filtered = _cluster_and_merge_lines(lines, blurred)
    # _draw_lines(blurred, lines_filtered, title="filtered lines")
    # intersections = _find_intersections(lines)
    # _draw_intersections(aligned_image, intersections)
    # cells = _extract_cells(aligned_image, lines_filtered)
    return []


def _extract_squares(image: np.ndarray) -> list[np.ndarray]:
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(image, -1, sharpen_kernel)
    save_debug_image(sharpen, title="sharpen")
    # Threshold and morph close
    thresh = cv2.threshold(sharpen, 50, 255, cv2.THRESH_BINARY_INV)[1]
    save_debug_image(thresh, title="thresh")
    morph_kernel_size = 5
    morph_shape = (morph_kernel_size, morph_kernel_size)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_shape)
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel=morph_kernel, iterations=2)
    save_debug_image(close, title="close")
    # Find contours and filter using threshold area
    contours = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    min_area = 100
    max_area = 150000
    count = 0
    squares = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(c)
            rect = image[y : y + h, x : x + w]
            save_debug_image(rect, title=f"square_{count}")
            squares.append(rect)
            count += 1
    # draw_squares(image, squares)
    return squares


def _cluster_and_merge_lines(lines: list[Line], image: np.ndarray | None) -> list[Line]:
    from sklearn.cluster import KMeans

    # Cluster lines offset using k-means
    rhos = np.array([line.rho for line in lines])
    rhos = rhos.reshape(-1, 1)
    kmeans = KMeans(n_clusters=15, random_state=0).fit(rhos)
    centers = kmeans.cluster_centers_
    # For each line, find the closest cluster center
    clusters = {}
    for line in lines:
        distances = np.linalg.norm(centers - np.array([line.rho, line.theta]), axis=1)
        closest_cluster = np.argmin(distances)
        if closest_cluster not in clusters:
            clusters[closest_cluster] = []
        clusters[closest_cluster].append(line)
    # Merge lines in each cluster
    merged_lines = []
    for cluster in clusters.values():
        rhos = [line.rho for line in cluster]
        thetas = [line.theta for line in cluster]
        rho = float(np.mean(rhos))
        theta = float(np.mean(thetas))
        line = Line(rho, theta)
        merged_lines.append(line)
    return merged_lines


# def _find_intersections(lines: Lines) -> list[np.ndarray]:
#     intersections = []
#     for h_line in lines.horizontal:
#         for v_line in lines.vertical:
#             rho1, theta1 = h_line
#             rho2, theta2 = v_line
#             A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
#             b = np.array([rho1, rho2])
#             x0, y0 = np.linalg.solve(A, b)
#             intersections.append(np.array([x0, y0]))
#     return intersections


def _draw_intersections(image: np.ndarray, intersections: list[np.ndarray]) -> np.ndarray:
    for intersection in intersections:
        x, y = intersection
        cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
    save_debug_image(image, title="intersections")
    return image


def _extract_cells(image: np.ndarray, lines: list[Line]) -> list[list[np.ndarray]]:
    cells = []
    for i in range(len(lines.horizontal) - 1):
        row_cells = []
        for j in range(len(lines.vertical) - 1):
            x1 = lines.vertical[j].rho
            x2 = lines.vertical[j + 1].rho
            y1 = lines.horizontal[i].rho
            y2 = lines.horizontal[i + 1].rho
            cell = image[int(y1) : int(y2), int(x1) : int(x2)]

            if _is_cell_square(cell):
                row_cells.append(cell)
                # save_debug_image(cell, f"cell_{i}_{j}")

        cells.append(row_cells)
    return cells


def _is_cell_square(cell: np.ndarray) -> bool:
    """
    Checks if a cell is square by comparing the aspect ratio to a threshold.

    Args:
        cell (np.ndarray): The cell image.

    Returns:
        bool: True if the cell is square, False otherwise.
    """
    height, width = cell.shape[:2]
    if height == 0 or width == 0:
        return False
    aspect_ratio = width / height
    return 0.7 <= aspect_ratio <= 1.4
