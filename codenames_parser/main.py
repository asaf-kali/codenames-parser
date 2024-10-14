import logging
import sys

from codenames.game.color import CardColor

from codenames_parser.align import align_image
from codenames_parser.color_detection import classify_cell_colors
from codenames_parser.crop import crop_image
from codenames_parser.grid_detection import extract_cells
from codenames_parser.image_reader import read_image
from codenames_parser.models import Grid
from scale import scale_down_image


def main(image_path: str) -> Grid[CardColor]:
    image = read_image(image_path)
    small_image = scale_down_image(image)
    aligned_image = align_image(small_image)
    cropped = crop_image(aligned_image)
    cells = extract_cells(cropped)
    grid_colors = classify_cell_colors(cells)
    return grid_colors


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", stream=sys.stdout)


def entrypoint():
    configure_logging()
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    grid = main(image_path)
    for row in grid:
        for cell in row:
            print(cell.emoji, end=" ")


if __name__ == "__main__":
    entrypoint()
