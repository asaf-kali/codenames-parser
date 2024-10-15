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

CASE_HIGH_RES = MapTestCase(
    fixture_file="high_res.png",
    expected_grid=Grid.from_rows(
        [
            [CardColor.GRAY, CardColor.RED, CardColor.BLUE, CardColor.GRAY, CardColor.BLUE],
            [CardColor.RED, CardColor.RED, CardColor.BLUE, CardColor.RED, CardColor.RED],
            [CardColor.BLUE, CardColor.BLUE, CardColor.GRAY, CardColor.GRAY, CardColor.GRAY],
            [CardColor.BLUE, CardColor.GRAY, CardColor.BLACK, CardColor.RED, CardColor.BLUE],
            [CardColor.RED, CardColor.BLUE, CardColor.GRAY, CardColor.RED, CardColor.RED],
        ]
    ),
)

CASE_SMALL_1 = MapTestCase(
    fixture_file="small_1.png",
    expected_grid=Grid.from_rows(
        [
            [CardColor.RED, CardColor.GRAY, CardColor.BLUE, CardColor.RED, CardColor.GRAY],
            [CardColor.BLUE, CardColor.BLACK, CardColor.BLUE, CardColor.GRAY, CardColor.BLUE],
            [CardColor.GRAY, CardColor.RED, CardColor.GRAY, CardColor.BLUE, CardColor.BLUE],
            [CardColor.RED, CardColor.BLUE, CardColor.RED, CardColor.RED, CardColor.GRAY],
            [CardColor.BLUE, CardColor.BLUE, CardColor.RED, CardColor.GRAY, CardColor.RED],
        ]
    ),
)

CASE_SMALL_2 = MapTestCase(
    fixture_file="small_2.png",
    expected_grid=Grid.from_rows(
        [
            [CardColor.RED, CardColor.BLUE, CardColor.GRAY, CardColor.BLUE, CardColor.GRAY],
            [CardColor.GRAY, CardColor.BLUE, CardColor.RED, CardColor.RED, CardColor.BLUE],
            [CardColor.GRAY, CardColor.RED, CardColor.GRAY, CardColor.BLUE, CardColor.RED],
            [CardColor.RED, CardColor.BLUE, CardColor.BLUE, CardColor.BLACK, CardColor.GRAY],
            [CardColor.RED, CardColor.GRAY, CardColor.RED, CardColor.BLUE, CardColor.RED],
        ]
    ),
)

MAP_CASES = [CASE_CUT_ROTATED, CASE_HIGH_RES, CASE_SMALL_1]
