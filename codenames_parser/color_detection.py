import logging

import numpy as np
from codenames.game.color import CardColor

from codenames_parser.consts import CARD_COLOR_TO_COLOR
from codenames_parser.debugging.util import SEPARATOR
from codenames_parser.models import Grid

log = logging.getLogger(__name__)


def classify_cell_colors(cells: Grid[np.ndarray]) -> Grid[CardColor]:
    """
    Classifies the color of each cell.

    Args:
        cells (list[list[np.ndarray]]): A 2D list of cell images.

    Returns:
        list[list[str]]: A 2D list of color names corresponding to each cell.
    """
    log.info(SEPARATOR)
    log.info("Classifying cell colors...")
    card_colors = Grid(row_size=cells.row_size)
    for cell_row in cells:
        row_colors = [detect_dominant_color(cell) for cell in cell_row]
        card_colors.append(row_colors)
    return card_colors


def detect_dominant_color(cell: np.ndarray) -> CardColor:
    """
    Detects the dominant color in a cell:
    """
    avg_color = cell.mean(axis=(0, 1))
    distances = {
        card_color: np.linalg.norm(avg_color - color.vector) for card_color, color in CARD_COLOR_TO_COLOR.items()
    }
    closest_color = min(distances, key=distances.get)
    return closest_color
