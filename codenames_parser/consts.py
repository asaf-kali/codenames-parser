from codenames.game.color import CardColor

from codenames_parser.models import Color

CODENAMES_RED = Color(r=250, g=45, b=45)
CODENAMES_BLUE = Color(r=45, g=115, b=170)
CODENAMES_YELLOW = Color(r=240, g=215, b=170)
CODENAMES_BLACK = Color(r=10, g=10, b=10)
CODENAMES_COLORS = [CODENAMES_RED, CODENAMES_BLUE, CODENAMES_YELLOW]

COLOR_TO_CARD_COLOR: dict[Color, CardColor] = {
    CODENAMES_RED: CardColor.RED,
    CODENAMES_BLUE: CardColor.BLUE,
    CODENAMES_YELLOW: CardColor.GRAY,
    CODENAMES_BLACK: CardColor.BLACK,
}

CARD_COLOR_TO_COLOR: dict[CardColor, Color] = {v: k for k, v in COLOR_TO_CARD_COLOR.items()}
