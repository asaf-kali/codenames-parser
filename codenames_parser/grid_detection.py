from typing import List

import numpy as np


def extract_cells(image: np.ndarray) -> List[List[np.ndarray]]:
    """
    Extracts the individual cells from the perspective-corrected image.

    Args:
        image (np.ndarray): The perspective-corrected image.

    Returns:
        List[np.ndarray]: A list of cell images.
    """
    grid_size = 5
    cells = []
    image_height, image_width = image.shape[:2]

    cell_height = image_height // grid_size
    cell_width = image_width // grid_size

    for row in range(grid_size):
        row_cells = []
        for col in range(grid_size):
            x_start = col * cell_width
            y_start = row * cell_height
            cell = image[y_start : y_start + cell_height, x_start : x_start + cell_width]
            row_cells.append(cell)
        cells.append(row_cells)

    return cells
