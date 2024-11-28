import sys
from dataclasses import dataclass

from codenames.classic.color import ClassicColor
from codenames.duet.card import DuetColor
from codenames.generic.card import CardColor
from codenames.utils.game_type import GameType

from codenames_parser.color_map.color_map_parser import parse_color_map
from codenames_parser.common.impr.grid_detection import GRID_WIDTH
from codenames_parser.common.impr.reader import read_image
from codenames_parser.common.utils.logging import configure_logging


@dataclass
class ParseColorMapArgs:
    image_path: str
    color_type: type[CardColor]


def entrypoint() -> list[CardColor]:
    configure_logging()
    # Parse arguments
    args = _parse_args()
    # Parse color map
    image = read_image(args.image_path)
    map_colors = parse_color_map(image, color_type=args.color_type)
    # Print result
    _print_result(map_colors)
    return map_colors


def _parse_args() -> ParseColorMapArgs:
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <image_path> <color_type>")
        sys.exit(1)
    image_path = sys.argv[1]
    game_type = sys.argv[2]
    color_type = _parse_color_type(game_type)
    return ParseColorMapArgs(image_path=image_path, color_type=color_type)


def _parse_color_type(game_type_arg: str) -> type[CardColor]:
    game_type = GameType[game_type_arg.upper()]
    if game_type == GameType.CLASSIC:
        return ClassicColor
    if game_type == GameType.DUET:
        return DuetColor
    raise ValueError(f"Invalid color type: {game_type_arg}")


def _print_result(map_colors: list[CardColor]):
    for i, color in enumerate(map_colors):
        if i % GRID_WIDTH == 0:
            print()
        print(color.emoji, end=" ")


if __name__ == "__main__":
    entrypoint()
