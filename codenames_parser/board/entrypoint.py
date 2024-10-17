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
    if len(sys.argv) > 2:
        language = sys.argv[2]
    else:
        language = "heb"
    board = parse_board(image_path, language=language)
    table = board.as_table
    print(table)


if __name__ == "__main__":
    entrypoint()
