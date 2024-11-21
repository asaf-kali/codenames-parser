from typing import Type

import numpy as np
from codenames.generic.card import CardColor

from codenames_parser.color_map.color_detection import classify_cell_colors
from codenames_parser.color_map.grid_detection import extract_cells
from codenames_parser.common.align import align_image
from codenames_parser.common.crop import crop_image
from codenames_parser.common.scale import scale_down_image


def parse_color_map[C: CardColor](image: np.ndarray, color_type: Type[C]) -> list[C]:
    scale_result = scale_down_image(image)
    alignment_result = align_image(scale_result.image)
    cropped = crop_image(alignment_result.aligned_image)
    cells = extract_cells(cropped, color_type=color_type)
    grid_colors = classify_cell_colors(cells, color_type=color_type)
    return grid_colors
