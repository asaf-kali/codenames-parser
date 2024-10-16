from codenames.game.color import CardColor

from codenames_parser.common.models import Color

CODENAMES_BLUE = Color(r=45, g=115, b=170)
CODENAMES_RED = Color(r=250, g=45, b=45)
CODENAMES_GRAY = Color(r=240, g=215, b=170)
CODENAMES_BLACK = Color(r=10, g=10, b=10)
CODENAMES_COLORS = [CODENAMES_BLUE, CODENAMES_RED, CODENAMES_GRAY]

COLOR_TO_CARD_COLOR: dict[Color, CardColor] = {
    CODENAMES_BLUE: CardColor.BLUE,
    CODENAMES_RED: CardColor.RED,
    CODENAMES_GRAY: CardColor.GRAY,
    CODENAMES_BLACK: CardColor.BLACK,
}

CARD_COLOR_TO_COLOR: dict[CardColor, Color] = {v: k for k, v in COLOR_TO_CARD_COLOR.items()}
