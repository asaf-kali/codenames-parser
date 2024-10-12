import cv2
import numpy as np


def classify_cell_colors(cells: list[list[np.ndarray]]) -> list[list[str]]:
    """
    Classifies the color of each cell.

    Args:
        cells (list[list[np.ndarray]]): A 2D list of cell images.

    Returns:
        list[list[str]]: A 2D list of color names corresponding to each cell.
    """
    color_names = []
    for row_cells in cells:
        row_colors = []
        for cell in row_cells:
            color = detect_dominant_color(cell)
            row_colors.append(color)
        color_names.append(row_colors)
    return color_names


def detect_dominant_color(cell: np.ndarray) -> str:
    """
    Detects the dominant color in a cell:
    1.
    Args:
        cell (np.ndarray): The cell image.

    Returns:
        str: The name of the dominant color.
    """
    # Resize cell to reduce computation
    small_cell = cv2.resize(cell, (50, 50))
    # Convert to HSV color space
    hsv_cell = cv2.cvtColor(small_cell, cv2.COLOR_BGR2HSV)
    # Compute histogram
    hist = cv2.calcHist([hsv_cell], [0], None, [180], [0, 180])
    dominant_hue = np.argmax(hist)

    # Define color ranges
    color_ranges = {
        "red": [(0, 10), (160, 180)],
        "yellow": [(20, 30)],
        "blue": [(100, 130)],
        "black": [(0, 180)],  # Assuming black has low saturation and value
    }

    # Average saturation and value to detect black
    avg_saturation = hsv_cell[:, :, 1].mean()
    avg_value = hsv_cell[:, :, 2].mean()
    if avg_value < 30 and avg_saturation < 30:
        return "black"

    for color_name, ranges in color_ranges.items():
        for lower, upper in ranges:
            if lower <= dominant_hue <= upper:
                return color_name

    return "unknown"
