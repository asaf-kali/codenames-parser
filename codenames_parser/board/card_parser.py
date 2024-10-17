import logging

import numpy as np
import pytesseract
from codenames.game.card import Card

from codenames_parser.board.ocr import fetch_tesseract_language
from codenames_parser.common.align import align_image, apply_rotations
from codenames_parser.common.debug_util import SEPARATOR, set_debug_context
from codenames_parser.common.scale import scale_down_image

log = logging.getLogger(__name__)

COMMON_ALPHABET = "-"
LANGUAGE_DEFAULT_ALPHABET = {
    "heb": "אבגדהוזחטיכלמנסעפצקרשתךםןףץ",
}


def parse_cards(cells: list[np.ndarray], language: str) -> list[Card]:
    cards = []
    for i, cell in enumerate(cells):
        set_debug_context(f"card {i}")
        log.info(SEPARATOR)
        log.info(f"Processing card {i}")
        card = _parse_card(cell, language=language)
        cards.append(card)
    return cards


def _parse_card(image: np.ndarray, language: str) -> Card:
    scale_result = scale_down_image(image, max_dimension=250)
    alignment_result = align_image(scale_result.image)
    image_aligned = apply_rotations(image=image, rotations=alignment_result.rotations)
    text = _extract_text(image_aligned, language=language)
    return Card(word=text.strip())


def _extract_text(card: np.ndarray, language: str) -> str:
    fetch_tesseract_language(language)
    config = "--psm 11"
    text = pytesseract.image_to_string(card, lang=language, config=config)
    text = text.replace("\n", " ")
    log.info(f"Extracted text: [{text}]")
    words = text.split()
    if not words:
        log.warning("No words extracted")
        return ""
    longest_word = str(max(words, key=len))
    log.info(f"Longest word: [{longest_word}]")
    return longest_word


# def _get_alphabet_option(language: str) -> str:
#     alphabet = LANGUAGE_DEFAULT_ALPHABET.get(language, None)
#     if alphabet is None:
#         return ""
#     alphabet += COMMON_ALPHABET
#     alphabet_option = f"-c tessedit_char_whitelist={alphabet}"
#     return alphabet_option
