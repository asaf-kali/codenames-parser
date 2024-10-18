import logging

import numpy as np
import pytesseract
from codenames.game.card import Card

from codenames_parser.board.ocr import fetch_tesseract_language
from codenames_parser.common.align import align_image, apply_rotations
from codenames_parser.common.debug_util import SEPARATOR, draw_boxes, set_debug_context
from codenames_parser.common.models import LetterBox
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
    # boxes = pytesseract.image_to_boxes(image_aligned, lang=language)
    # color_distance = color_distance_mask(alignment_result.aligned_image, color=WHITE, percentile=80)
    text = _extract_text(image_aligned, language=language)
    return Card(word=text.strip())


def _extract_text(card: np.ndarray, language: str) -> str:
    fetch_tesseract_language(language)
    config = "--psm 11"
    result = pytesseract.image_to_boxes(card, lang=language, config=config)
    letter_boxes = _parse_image_to_boxes(result)
    letter_boxes_mirrored = _mirror_around_horizontal_center(card, letter_boxes)
    draw_boxes(image=card, boxes=letter_boxes_mirrored, title="letter boxes")
    word = _find_word(letter_boxes)
    return word


def _mirror_around_horizontal_center(image: np.ndarray, boxes: list[LetterBox]) -> list[LetterBox]:
    image_height = image.shape[0]
    center_y = image_height // 2
    mirrored_boxes = []
    for box in boxes:
        y1 = center_y - (box.y + box.h - center_y)
        mirrored_box = LetterBox(x=box.x, y=y1, w=box.w, h=box.h, letter=box.letter)
        mirrored_boxes.append(mirrored_box)
    return mirrored_boxes


def _pick_word_from_raw_text(raw_text: str) -> str:
    log.info(f"Extracted text: [{raw_text}]")
    words = raw_text.split()
    if not words:
        log.warning("No words extracted")
        return ""
    longest_word = str(max(words, key=len))
    log.info(f"Longest word: [{longest_word}]")
    return longest_word


def _find_word(boxes: list[LetterBox]) -> str:
    text = "".join(box.letter for box in boxes)
    return text


def _parse_image_to_boxes(result: str) -> list[LetterBox]:
    boxes_raw = result.split("\n")
    if not boxes_raw:
        return []
    boxes = []
    for box_raw in boxes_raw:
        box = _parse_letter_box(box_raw)
        if not box:
            continue
        boxes.append(box)
    return boxes


def _parse_letter_box(letter_box: str) -> LetterBox | None:
    parts = letter_box.split(" ")
    if len(parts) != 6:
        return None
    letter = parts[0]
    x1, y1, x2, y2 = map(int, parts[1:5])
    w = x2 - x1
    h = y2 - y1
    return LetterBox(x=x1, y=y1, w=w, h=h, letter=letter)


# def _get_alphabet_option(language: str) -> str:
#     alphabet = LANGUAGE_DEFAULT_ALPHABET.get(language, None)
#     if alphabet is None:
#         return ""
#     alphabet += COMMON_ALPHABET
#     alphabet_option = f"-c tessedit_char_whitelist={alphabet}"
#     return alphabet_option
