import cv2
import numpy as np

from codenames_parser.debugging.util import save_debug_image
from codenames_parser.models import P1P2, Color, Line, Point


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
    blurred = _blur_image(image)
    edges = _detect_edges(blurred)
    lines = _extract_lines(edges)
    _draw_lines(blurred, lines, title="lines")
    lines_filtered = _cluster_and_merge_lines(lines)
    _draw_lines(blurred, lines_filtered, title="filtered lines")
    aligned_image = _align_image(image, lines_filtered)
    # intersections = _find_intersections(lines)
    # _draw_intersections(aligned_image, intersections)
    cells = _extract_cells(aligned_image, lines_filtered)
    return cells


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


def _extract_lines(edges: np.ndarray) -> list[Line]:
    # Find lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    _lines = []
    for line in lines:
        rho, theta = line[0]
        line = Line(rho, theta)
        _lines.append(line)
    return _lines


def _cluster_and_merge_lines(lines: list[Line]) -> list[Line]:
    # Cluster lines based on angle
    clusters = {}
    for line in lines:
        theta = line.theta
        for cluster_theta in clusters:
            if abs(theta - cluster_theta) < np.pi / 18:
                clusters[cluster_theta].append(line)
                break
        else:
            clusters[theta] = [line]

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


def _align_image(image: np.ndarray, lines: list[Line]) -> np.ndarray:
    # Rotate the image to align the grid lines
    horizontal_angle = np.mean([abs(line.theta - np.pi / 2) for line in lines.horizontal])
    vertical_angle = np.mean([min(line.theta, np.pi - line.theta) for line in lines.vertical])
    angle = (horizontal_angle + vertical_angle) / 2
    center = (image.shape[1] / 2, image.shape[0] / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle * 180 / np.pi, 1)
    aligned_image = cv2.warpAffine(image, rotation_matrix, image.shape[:2])
    save_debug_image(aligned_image, title="aligned")
    return aligned_image


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
