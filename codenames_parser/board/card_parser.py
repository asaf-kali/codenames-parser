import numpy as np
from codenames.game.card import Card
from codenames.game.color import CardColor

from codenames_parser.board.grid_detection import CARD_RATIO
from codenames_parser.common.align import detect_edges
from codenames_parser.common.debug_util import draw_boxes, set_debug_context
from codenames_parser.common.grid_detection import find_boxes


def parse_cards(cells: list[np.ndarray], language: str) -> list[Card]:
    cards = []
    for i, cell in enumerate(cells):
        set_debug_context(f"card {i}")
        card = _parse_card(cell, language=language)
        cards.append(card)
    return cards


def _parse_card(image: np.ndarray, language: str) -> Card:
    edges = detect_edges(image, threshold1=120, threshold2=240)
    # alignment_result = align_image(image)
    # cropped = crop_image(alignment_result.aligned_image)
    boxes = find_boxes(edges, expected_ratio=CARD_RATIO, max_ratio_diff=7)
    draw_boxes(edges, boxes=boxes, title="boxes")
    print(boxes, language)
    return Card(word="word", color=CardColor.BLACK)
