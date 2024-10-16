import numpy as np
from codenames.game.card import Card

from codenames_parser.common.models import Grid


def parse_cards(cells: Grid[np.ndarray]) -> Grid[Card]:
    grid: Grid[Card] = Grid(row_size=len(cells))
    return grid
