# pylint: disable=R0801
import sys

from codenames.game.board import Board
from codenames.game.card import Card

from codenames_parser.board.board_parser import parse_board
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
    words = parse_board(image_path, language=language)
    cards = [Card(word=word) for word in words]
    board = Board(cards=cards, language=language)
    table = board.as_table
    print(table)


if __name__ == "__main__":
    entrypoint()
