from typing import NamedTuple

from codenames.classic.color import ClassicColor


class ColorMapTestCase(NamedTuple):
    fixture_file: str
    expected_colors: list[ClassicColor]


COLORS_1 = [
    ClassicColor.NEUTRAL,
    ClassicColor.RED,
    ClassicColor.BLUE,
    ClassicColor.BLUE,
    ClassicColor.RED,
    ClassicColor.NEUTRAL,
    ClassicColor.BLUE,
    ClassicColor.RED,
    ClassicColor.NEUTRAL,
    ClassicColor.BLUE,
    ClassicColor.BLUE,
    ClassicColor.RED,
    ClassicColor.RED,
    ClassicColor.BLUE,
    ClassicColor.NEUTRAL,
    ClassicColor.BLUE,
    ClassicColor.RED,
    ClassicColor.RED,
    ClassicColor.NEUTRAL,
    ClassicColor.NEUTRAL,
    ClassicColor.NEUTRAL,
    ClassicColor.BLUE,
    ClassicColor.ASSASSIN,
    ClassicColor.RED,
    ClassicColor.BLUE,
]


COLORS_2 = [
    ClassicColor.RED,
    ClassicColor.NEUTRAL,
    ClassicColor.BLUE,
    ClassicColor.RED,
    ClassicColor.NEUTRAL,
    ClassicColor.BLUE,
    ClassicColor.ASSASSIN,
    ClassicColor.BLUE,
    ClassicColor.NEUTRAL,
    ClassicColor.BLUE,
    ClassicColor.NEUTRAL,
    ClassicColor.RED,
    ClassicColor.NEUTRAL,
    ClassicColor.BLUE,
    ClassicColor.BLUE,
    ClassicColor.RED,
    ClassicColor.BLUE,
    ClassicColor.RED,
    ClassicColor.RED,
    ClassicColor.NEUTRAL,
    ClassicColor.BLUE,
    ClassicColor.BLUE,
    ClassicColor.RED,
    ClassicColor.NEUTRAL,
    ClassicColor.RED,
]


COLORS_3 = [
    ClassicColor.NEUTRAL,
    ClassicColor.RED,
    ClassicColor.BLUE,
    ClassicColor.NEUTRAL,
    ClassicColor.BLUE,
    ClassicColor.RED,
    ClassicColor.RED,
    ClassicColor.BLUE,
    ClassicColor.RED,
    ClassicColor.RED,
    ClassicColor.BLUE,
    ClassicColor.BLUE,
    ClassicColor.NEUTRAL,
    ClassicColor.NEUTRAL,
    ClassicColor.NEUTRAL,
    ClassicColor.BLUE,
    ClassicColor.NEUTRAL,
    ClassicColor.ASSASSIN,
    ClassicColor.RED,
    ClassicColor.BLUE,
    ClassicColor.RED,
    ClassicColor.BLUE,
    ClassicColor.NEUTRAL,
    ClassicColor.RED,
    ClassicColor.RED,
]


CASE_TOP_VIEW = ColorMapTestCase(fixture_file="top_view.png", expected_colors=COLORS_1)
CASE_CUT_ROTATED = ColorMapTestCase(fixture_file="cut_rotated.png", expected_colors=COLORS_1)
CASE_SMALL_1 = ColorMapTestCase(fixture_file="small_1.png", expected_colors=COLORS_2)
CASE_HIGH_RES = ColorMapTestCase(fixture_file="high_res.png", expected_colors=COLORS_3)

MAP_CASES = [CASE_TOP_VIEW, CASE_CUT_ROTATED, CASE_SMALL_1, CASE_HIGH_RES]
