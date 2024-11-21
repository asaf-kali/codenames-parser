from typing import NamedTuple

from codenames.classic.color import ClassicColor
from codenames.duet.card import DuetColor
from codenames.generic.card import CardColor


class ColorMapTestCase(NamedTuple):
    fixture_file: str
    expected_colors: list[ClassicColor]
    color_type: type[CardColor]


### Classic ###

CLASSIC_COLORS_1 = [
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
CLASSIC_COLORS_2 = [
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
CLASSIC_COLORS_3 = [
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

CLASSIC_TOP = ColorMapTestCase(fixture_file="top_view.png", expected_colors=CLASSIC_COLORS_1, color_type=ClassicColor)
CLASSIC_ROTATED = ColorMapTestCase(
    fixture_file="cut_rotated.png", expected_colors=CLASSIC_COLORS_1, color_type=ClassicColor
)
CLASSIC_SMALL_1 = ColorMapTestCase(
    fixture_file="small_1.png", expected_colors=CLASSIC_COLORS_2, color_type=ClassicColor
)
CLASSIC_HIGH_RES = ColorMapTestCase(
    fixture_file="high_res.png", expected_colors=CLASSIC_COLORS_3, color_type=ClassicColor
)

### Duet ###

DUET_COLORS_1 = [
    # Row 1
    DuetColor.NEUTRAL,
    DuetColor.ASSASSIN,
    DuetColor.NEUTRAL,
    DuetColor.NEUTRAL,
    DuetColor.NEUTRAL,
    # Row 2
    DuetColor.NEUTRAL,
    DuetColor.GREEN,
    DuetColor.NEUTRAL,
    DuetColor.GREEN,
    DuetColor.NEUTRAL,
    # Row 3
    DuetColor.ASSASSIN,
    DuetColor.NEUTRAL,
    DuetColor.GREEN,
    DuetColor.GREEN,
    DuetColor.GREEN,
    # Row 4
    DuetColor.NEUTRAL,
    DuetColor.GREEN,
    DuetColor.ASSASSIN,
    DuetColor.GREEN,
    DuetColor.NEUTRAL,
    # Row 1
    DuetColor.NEUTRAL,
    DuetColor.NEUTRAL,
    DuetColor.GREEN,
    DuetColor.NEUTRAL,
    DuetColor.GREEN,
]

DUET_COLORS_2 = [
    # Row 1
    DuetColor.NEUTRAL,
    DuetColor.GREEN,
    DuetColor.NEUTRAL,
    DuetColor.NEUTRAL,
    DuetColor.ASSASSIN,
    # Row 2
    DuetColor.NEUTRAL,
    DuetColor.NEUTRAL,
    DuetColor.NEUTRAL,
    DuetColor.GREEN,
    DuetColor.GREEN,
    # Row 3
    DuetColor.GREEN,
    DuetColor.NEUTRAL,
    DuetColor.NEUTRAL,
    DuetColor.NEUTRAL,
    DuetColor.NEUTRAL,
    # Row 4
    DuetColor.GREEN,
    DuetColor.GREEN,
    DuetColor.GREEN,
    DuetColor.ASSASSIN,
    DuetColor.NEUTRAL,
    # Row 5
    DuetColor.NEUTRAL,
    DuetColor.ASSASSIN,
    DuetColor.GREEN,
    DuetColor.GREEN,
    DuetColor.NEUTRAL,
]

DUET_MAP1_FRONT = ColorMapTestCase(fixture_file="map1_front.jpg", expected_colors=DUET_COLORS_1, color_type=DuetColor)
DUET_MAP1_TOP = ColorMapTestCase(fixture_file="map1_top.jpg", expected_colors=DUET_COLORS_1, color_type=DuetColor)
DUET_MAP2_FRONT = ColorMapTestCase(fixture_file="map2_front.jpg", expected_colors=DUET_COLORS_2, color_type=DuetColor)
DUET_MAP2_TOP = ColorMapTestCase(fixture_file="map2_top.jpg", expected_colors=DUET_COLORS_2, color_type=DuetColor)

MAP_CASES = [
    # Classic
    CLASSIC_TOP,
    CLASSIC_ROTATED,
    CLASSIC_SMALL_1,
    CLASSIC_HIGH_RES,
    # Duet
    DUET_MAP1_FRONT,
    DUET_MAP1_TOP,
    DUET_MAP2_FRONT,
    DUET_MAP2_TOP,
]
