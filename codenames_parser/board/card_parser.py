import logging

import numpy as np
import pytesseract
from codenames.game.card import Card

from codenames_parser.board.ocr import fetch_tesseract_language
from codenames_parser.common.align import align_image
from codenames_parser.common.debug_util import set_debug_context
from codenames_parser.common.scale import scale_down_image

log = logging.getLogger(__name__)


def parse_cards(cells: list[np.ndarray], language: str) -> list[Card]:
    cards = []
    for i, cell in enumerate(cells):
        set_debug_context(f"card {i}")
        card = _parse_card(cell, language=language)
        cards.append(card)
    return cards


def _parse_card(image: np.ndarray, language: str) -> Card:
    scale_result = scale_down_image(image, max_dimension=250)
    alignment_result = align_image(scale_result.image)
    fetch_tesseract_language(language)
    text = pytesseract.image_to_string(alignment_result.aligned_image, lang=language, config="--psm 6")
    log.info(f"Extracted text: '{text}'")
    return Card(word=text.strip())
