from typing import NamedTuple

from codenames.game.color import CardColor


class ColorMapTestCase(NamedTuple):
    fixture_file: str
    expected_colors: list[CardColor]


COLORS_1 = [
    CardColor.GRAY,
    CardColor.RED,
    CardColor.BLUE,
    CardColor.BLUE,
    CardColor.RED,
    CardColor.GRAY,
    CardColor.BLUE,
    CardColor.RED,
    CardColor.GRAY,
    CardColor.BLUE,
    CardColor.BLUE,
    CardColor.RED,
    CardColor.RED,
    CardColor.BLUE,
    CardColor.GRAY,
    CardColor.BLUE,
    CardColor.RED,
    CardColor.RED,
    CardColor.GRAY,
    CardColor.GRAY,
    CardColor.GRAY,
    CardColor.BLUE,
    CardColor.BLACK,
    CardColor.RED,
    CardColor.BLUE,
]


COLORS_2 = [
    CardColor.RED,
    CardColor.GRAY,
    CardColor.BLUE,
    CardColor.RED,
    CardColor.GRAY,
    CardColor.BLUE,
    CardColor.BLACK,
    CardColor.BLUE,
    CardColor.GRAY,
    CardColor.BLUE,
    CardColor.GRAY,
    CardColor.RED,
    CardColor.GRAY,
    CardColor.BLUE,
    CardColor.BLUE,
    CardColor.RED,
    CardColor.BLUE,
    CardColor.RED,
    CardColor.RED,
    CardColor.GRAY,
    CardColor.BLUE,
    CardColor.BLUE,
    CardColor.RED,
    CardColor.GRAY,
    CardColor.RED,
]


COLORS_3 = [
    CardColor.GRAY,
    CardColor.RED,
    CardColor.BLUE,
    CardColor.GRAY,
    CardColor.BLUE,
    CardColor.RED,
    CardColor.RED,
    CardColor.BLUE,
    CardColor.RED,
    CardColor.RED,
    CardColor.BLUE,
    CardColor.BLUE,
    CardColor.GRAY,
    CardColor.GRAY,
    CardColor.GRAY,
    CardColor.BLUE,
    CardColor.GRAY,
    CardColor.BLACK,
    CardColor.RED,
    CardColor.BLUE,
    CardColor.RED,
    CardColor.BLUE,
    CardColor.GRAY,
    CardColor.RED,
    CardColor.RED,
]


CASE_TOP_VIEW = ColorMapTestCase(fixture_file="top_view.png", expected_colors=COLORS_1)
CASE_CUT_ROTATED = ColorMapTestCase(fixture_file="cut_rotated.png", expected_colors=COLORS_1)
CASE_SMALL_1 = ColorMapTestCase(fixture_file="small_1.png", expected_colors=COLORS_2)
CASE_HIGH_RES = ColorMapTestCase(fixture_file="high_res.png", expected_colors=COLORS_3)

MAP_CASES = [CASE_TOP_VIEW, CASE_CUT_ROTATED, CASE_SMALL_1, CASE_HIGH_RES]
