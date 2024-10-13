import logging
from typing import NamedTuple

import cv2
import numpy as np

from codenames_parser.debugging.util import save_debug_image
from codenames_parser.models import P1P2, Color, GridLines, Line, Point

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
    aligned_image = _align_image(image)
    blurred = _blur_image(aligned_image)
    edges = _detect_edges(blurred)
    lines = _extract_lines(edges, rho=0.2)
    _draw_lines(aligned_image, lines, title="lines_after_alignment")
    # lines_filtered = _cluster_and_merge_lines(lines, blurred)
    # _draw_lines(blurred, lines_filtered, title="filtered lines")
    # intersections = _find_intersections(lines)
    # _draw_intersections(aligned_image, intersections)
    # cells = _extract_cells(aligned_image, lines_filtered)
    return []


def _blur_image(image: np.ndarray) -> np.ndarray:
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    save_debug_image(blurred, title="blurred")
    return blurred


def _detect_edges(image: np.ndarray) -> np.ndarray:
    # Edge detection
    edges = cv2.Canny(image, 50, 150)
    save_debug_image(edges, title="edges")
    return edges


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
    _draw_squares(image, squares)
    return squares


def _extract_lines(edges: np.ndarray, rho: float = 1, theta: float = np.pi / 180, threshold: int = 100) -> list[Line]:
    # Find lines using Hough transform
    hough_lines = cv2.HoughLines(edges, rho=rho, theta=theta, threshold=threshold)
    if hough_lines is None:
        return []
    lines = []
    for line in hough_lines:
        rho, theta = line[0]
        line = Line(rho, theta)
        lines.append(line)
    return lines


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


def _draw_lines(image: np.ndarray, lines: list[Line], title: str) -> np.ndarray:
    # If image is grayscale, convert to BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for line in lines:
        loc = _get_line_draw_params(line)
        color = _pick_line_color(line)
        cv2.line(image, loc.p1, loc.p2, color, 2)
    save_debug_image(image, title=title)
    return image


def _draw_squares(image: np.ndarray, squares: list[np.ndarray]) -> np.ndarray:
    for square in squares:
        cv2.rectangle(image, (0, 0), (square.shape[1], square.shape[0]), (0, 255, 0), 2)
    save_debug_image(image, title="squares")
    return image


def _pick_line_color(line: Line) -> Color:
    red = np.sin(line.theta)
    blue = 1 - red
    color = 200 * np.array([blue, 0, red])
    random_offset = np.random.randint(0, 50, 3)
    color += random_offset
    rounded = np.round(color)
    return Color(*rounded.tolist())


def _get_line_draw_params(line: Line) -> P1P2:
    SIZE = 1000
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + SIZE * (-b))
    x2 = int(x0 - SIZE * (-b))
    y1 = int(y0 + SIZE * (a))
    y2 = int(y0 - SIZE * (a))
    p1 = Point(x1, y1)
    p2 = Point(x2, y2)
    return P1P2(p1, p2)


def _align_image(image: np.ndarray) -> np.ndarray:
    rho = 1
    while True:
        result = _align_image_iteration(image, rho=rho)
        image = result.image
        if result.line_count < 10:
            break
        rho /= 1.5
    save_debug_image(image, title="aligned_final")
    return image


class AlignmentResult(NamedTuple):
    image: np.ndarray
    line_count: int


def _align_image_iteration(image: np.ndarray, rho: float) -> AlignmentResult:
    blurred = _blur_image(image)
    edges = _detect_edges(blurred)
    lines = _extract_lines(edges, rho=rho)
    if not lines:
        return AlignmentResult(image, 0)
    _draw_lines(blurred, lines, title="lines_before_rotate")
    angle_degrees = _find_rotation_angle(lines)
    log.info(f"Rotation angle: {angle_degrees}")
    aligned_image = _rotate_by(image, angle_degrees)
    save_debug_image(aligned_image, title="aligned")
    return AlignmentResult(aligned_image, len(lines))


def _rotate_by(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    # Get the image center and dimensions
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1)
    # Calculate the new bounding dimensions of the rotated image
    cos_angle = np.abs(rotation_matrix[0, 0])
    sin_angle = np.abs(rotation_matrix[0, 1])
    new_w = int(h * sin_angle + w * cos_angle)
    new_h = int(h * cos_angle + w * sin_angle)
    # Adjust the rotation matrix to take into account the translation
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    # Apply the warp affine with the new dimensions to preserve all pixels
    aligned_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
    return aligned_image


def _find_rotation_angle(lines: list[Line]) -> float:
    grid_lines = _get_grid_lines(lines)
    horizontal_angle = np.mean([abs(line.theta - np.pi / 2) for line in grid_lines.horizontal])
    vertical_angle = np.mean([min(line.theta, np.pi - line.theta) for line in grid_lines.vertical])
    angle = (horizontal_angle + vertical_angle) / 2
    angle_degrees = -np.degrees(angle)
    return angle_degrees


def _get_grid_lines(lines: list[Line]) -> GridLines:
    horizontal = []
    vertical = []
    for line in lines:
        if _is_vertical_line(line):
            vertical.append(line)
        else:
            horizontal.append(line)
    return GridLines(horizontal, vertical)


def _is_vertical_line(line: Line) -> bool:
    if line.theta < np.pi / 4 or line.theta > 3 * np.pi / 4:
        return True
    return False


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
