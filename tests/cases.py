from typing import NamedTuple

from codenames.game.color import CardColor

from codenames_parser.models import Grid


class MapTestCase(NamedTuple):
    fixture_file: str
    expected_grid: Grid[CardColor]


CASE_CUT_ROTATED = MapTestCase(
    fixture_file="cut_rotated.png",
    expected_grid=Grid.from_rows(
        [
            [CardColor.GRAY, CardColor.RED, CardColor.BLUE, CardColor.BLUE, CardColor.RED],
            [CardColor.GRAY, CardColor.BLUE, CardColor.RED, CardColor.GRAY, CardColor.BLUE],
            [CardColor.BLUE, CardColor.RED, CardColor.RED, CardColor.BLUE, CardColor.GRAY],
            [CardColor.BLUE, CardColor.RED, CardColor.RED, CardColor.GRAY, CardColor.GRAY],
            [CardColor.GRAY, CardColor.BLUE, CardColor.BLACK, CardColor.RED, CardColor.BLUE],
        ]
    ),
)
