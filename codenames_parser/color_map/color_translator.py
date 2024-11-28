from codenames.classic.color import ClassicColor
from codenames.duet.card import DuetColor
from codenames.generic.card import CardColor

from codenames_parser.common.utils.models import Color

CODENAMES_BLUE = Color(r=45, g=115, b=170)
CODENAMES_RED = Color(r=250, g=45, b=45)
CODENAMES_GREEN = Color(r=26, g=102, b=37)
CODENAMES_GRAY = Color(r=240, g=215, b=170)
CODENAMES_BLACK = Color(r=20, g=20, b=20)

CODENAMES_COLORS_CLASSIC = [CODENAMES_BLUE, CODENAMES_RED, CODENAMES_GRAY]
CODENAMES_COLORS_DUET = [CODENAMES_GREEN, CODENAMES_GRAY, CODENAMES_BLACK]

ColorTranslator = dict[CardColor, Color]

CARD_COLOR_TO_COLOR_CLASSIC: dict[ClassicColor, Color] = {
    ClassicColor.BLUE: CODENAMES_BLUE,
    ClassicColor.RED: CODENAMES_RED,
    ClassicColor.NEUTRAL: CODENAMES_GRAY,
    ClassicColor.ASSASSIN: CODENAMES_BLACK,
}

CARD_COLOR_TO_COLOR_DUET: dict[DuetColor, Color] = {
    DuetColor.GREEN: CODENAMES_GREEN,
    DuetColor.NEUTRAL: CODENAMES_GRAY,
    DuetColor.ASSASSIN: CODENAMES_BLACK,
}


def get_color_translator(color_type: type[CardColor]) -> ColorTranslator:
    if color_type == ClassicColor:
        return CARD_COLOR_TO_COLOR_CLASSIC
    if color_type == DuetColor:
        return CARD_COLOR_TO_COLOR_DUET
    raise NotImplementedError(f"Unsupported color type: {color_type}")


def get_board_colors(color_type: type[CardColor]) -> list[Color]:
    if color_type == ClassicColor:
        return CODENAMES_COLORS_CLASSIC
    if color_type == DuetColor:
        return CODENAMES_COLORS_DUET
    raise NotImplementedError(f"Unsupported color type: {color_type}")
