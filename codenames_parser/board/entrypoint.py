# pylint: disable=R0801
import sys

from codenames_parser.board.parser import parse_board
from codenames_parser.common.logging import configure_logging


def entrypoint():
    configure_logging()
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    grid = parse_board(image_path)
    for row in grid:
        for cell in row:
            print(cell.emoji, end=" ")
        print("")


if __name__ == "__main__":
    entrypoint()
