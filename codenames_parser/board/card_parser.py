import numpy as np
from codenames.game.card import Card
from codenames.game.color import CardColor

from codenames_parser.common.align import align_image
from codenames_parser.common.crop import crop_image
from codenames_parser.common.debug_util import set_debug_context
from codenames_parser.common.models import Grid


def parse_cards(cells: Grid[np.ndarray]) -> list[Card]:
    cards = []
    for i, cell in enumerate(cells):
        set_debug_context(f"card {i}")
        card = _parse_card(cell)
        cards.append(card)
    return cards


def _parse_card(image: np.ndarray) -> Card:
    alignment_result = align_image(image)
    # TODO: Aligned image is not used bug
    cropped = crop_image(alignment_result.aligned_image)

    return Card(word="word", color=CardColor.BLACK)
