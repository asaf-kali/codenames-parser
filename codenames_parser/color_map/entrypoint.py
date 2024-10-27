import sys

from codenames_parser.color_map.color_map_parser import parse_color_map
from codenames_parser.common.debug_util import set_save_debug_images
from codenames_parser.common.grid_detection import GRID_WIDTH
from codenames_parser.common.image_reader import read_image
from codenames_parser.common.logging import configure_logging


def entrypoint():
    configure_logging()
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    set_save_debug_images(enabled=True)
    image = read_image(image_path)
    map_colors = parse_color_map(image)
    for i, color in enumerate(map_colors):
        if i % GRID_WIDTH == 0:
            print()
        print(color.emoji, end=" ")


if __name__ == "__main__":
    entrypoint()
