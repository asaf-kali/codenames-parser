import sys

from codenames_parser.color_map.parser import parse_color_map
from codenames_parser.common.logging import configure_logging


def entrypoint():
    configure_logging()
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    grid = parse_color_map(image_path)
    for row in grid.rows:
        for cell in row:
            print(cell.emoji, end=" ")
        print("")


if __name__ == "__main__":
    entrypoint()
