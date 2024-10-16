from codenames.game.color import CardColor

from codenames_parser.color_map.color_detection import classify_cell_colors
from codenames_parser.color_map.grid_detection import extract_cells
from codenames_parser.common.align import align_image
from codenames_parser.common.crop import crop_image
from codenames_parser.common.image_reader import read_image
from codenames_parser.common.models import Grid
from codenames_parser.common.scale import scale_down_image


def parse_color_map(image_path: str) -> Grid[CardColor]:
    image = read_image(image_path)
    small_image = scale_down_image(image)
    aligned_image = align_image(small_image)
    cropped = crop_image(aligned_image)
    cells = extract_cells(cropped)
    grid_colors = classify_cell_colors(cells)
    return grid_colors
