import numpy as np
from codenames.game.card import Card
from codenames.game.color import CardColor

from codenames_parser.board.grid_detection import CARD_RATIO
from codenames_parser.common.align import align_image, detect_edges
from codenames_parser.common.debug_util import draw_boxes, set_debug_context
from codenames_parser.common.grid_detection import find_boxes
from codenames_parser.common.scale import scale_down_image


def parse_cards(cells: list[np.ndarray], language: str) -> list[Card]:
    cards = []
    for i, cell in enumerate(cells):
        set_debug_context(f"card {i}")
        card = _parse_card(cell, language=language)
        cards.append(card)
    return cards


def _parse_card(image: np.ndarray, language: str) -> Card:
    scale_result = scale_down_image(image, max_dimension=100)
    alignment_result = align_image(scale_result.image)
    edges = detect_edges(alignment_result.aligned_image, threshold1=50, threshold2=150)
    # cropped = crop_image(alignment_result.aligned_image)
    boxes = find_boxes(edges, expected_ratio=CARD_RATIO, max_ratio_diff=0.2)
    draw_boxes(edges, boxes=boxes, title="boxes")
    return Card(word=language, color=CardColor.BLACK)
