# pylint: disable=R0801
import sys
from dataclasses import dataclass

from codenames.generic.board import Board
from codenames.generic.card import Card

from codenames_parser.board.board_parser import parse_board
from codenames_parser.common.impr.reader import read_image
from codenames_parser.common.utils.logging import configure_logging


@dataclass
class ParseBoardArgs:
    image_path: str
    language: str


def entrypoint() -> list[str]:
    configure_logging()
    # Parse arguments
    args = _parse_args()
    image = read_image(args.image_path)
    # Parse board
    words = parse_board(image, language=args.language)
    # Print result
    _print_words(words=words)
    return words


def _parse_args() -> ParseBoardArgs:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <image_path> [<language>]")
        sys.exit(1)
    image_path = sys.argv[1]
    if len(sys.argv) > 2:
        language = sys.argv[2]
    else:
        language = "heb"
    return ParseBoardArgs(image_path=image_path, language=language)


def _print_words(words: list[str]):
    cards = [Card(word=word, color=None) for word in words]
    board = Board(cards=cards, language="")
    table = board.as_table
    print(table)


if __name__ == "__main__":
    entrypoint()
