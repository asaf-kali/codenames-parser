import logging

import numpy as np
import pytesseract
from codenames.game.card import Card

from codenames_parser.board.ocr import fetch_tesseract_language
from codenames_parser.common.align import align_image, apply_rotations
from codenames_parser.common.debug_util import (
    SEPARATOR,
    draw_boxes,
    save_debug_image,
    set_debug_context,
)
from codenames_parser.common.grid_detection import crop_cells, deduplicate_boxes
from codenames_parser.common.models import Box, LetterBox
from codenames_parser.common.scale import scale_down_image

log = logging.getLogger(__name__)


# COMMON_ALPHABET = "-"
# LANGUAGE_DEFAULT_ALPHABET = {
#     "heb": "אבגדהוזחטיכלמנסעפצקרשתךםןףץ",
# }


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
    result = pytesseract.image_to_string(card, lang=language, config=config)
    word = _pick_word_from_raw_text(result)
    log.info(f"Extracted word: [{word}]")
    return word


def _extract_text_boxes_strategy(card: np.ndarray, language: str) -> str:
    fetch_tesseract_language(language)
    config = "--psm 11"
    result = pytesseract.image_to_boxes(card, lang=language, config=config)
    letter_boxes_raw = _parse_letter_boxes(result)
    letter_boxes = _mirror_around_horizontal_center(card, letter_boxes_raw)
    draw_boxes(image=card, boxes=letter_boxes, title="letter boxes")
    letter_boxes_deduplicated = deduplicate_boxes(boxes=letter_boxes, max_iou=0.1)  # type: ignore[arg-type]
    draw_boxes(image=card, boxes=letter_boxes_deduplicated, title="letter boxes deduplicated")
    # _log_letter_boxes(card=card, letter_boxes=letter_boxes_deduplicated)
    max_letter_distance = _get_max_letter_distance(card.shape)
    word_boxes = _merge_letter_boxes(
        letter_boxes=letter_boxes_deduplicated,
        max_center_distance=max_letter_distance,
    )  # type: ignore[arg-type]
    draw_boxes(image=card, boxes=word_boxes, title="word boxes")
    word_cells = crop_cells(card, boxes=word_boxes)
    word = _find_word(word_cells, language=language)
    log.info(f"Extracted word: [{word}]")
    return word


def _log_letter_boxes(card: np.ndarray, letter_boxes: list[Box]) -> None:
    letters = crop_cells(card, boxes=letter_boxes)
    for i, letter in enumerate(letters):
        save_debug_image(letter, title=f"letter {i}")


def _get_max_letter_distance(image_shape: tuple[int, int]) -> float:
    image_width = image_shape[1]
    return image_width / 10


def _merge_letter_boxes(letter_boxes: list[Box], max_center_distance: float) -> list[Box]:
    """
    Merge letter boxes that are close to each other.
    Take min x, min y, max x, max y.
    The threshold determines how far apart letters can be to still be merged into a word box.
    """
    if not letter_boxes:
        return []
    max_side_distance = max_center_distance / 2
    # Distance matrix
    num_boxes = len(letter_boxes)
    distances = np.full((num_boxes, num_boxes), np.inf)
    for i, box1 in enumerate(letter_boxes):
        for j, box2 in enumerate(letter_boxes):
            if i <= j:
                continue
            distances[i, j] = box1.center_distance(box2)
    # Merge boxes
    merge_candidates_indices = np.argwhere(distances < max_center_distance)
    for i, j in merge_candidates_indices:
        box1, box2 = letter_boxes[i], letter_boxes[j]
        if id(box1) == id(box2):
            continue
        side_distance = _box_side_distance(box1, box2)
        if side_distance > max_side_distance:
            continue
        merged_x = min(box1.x, box2.x)
        merged_y = min(box1.y, box2.y)
        merged_w = max(box1.x + box1.w, box2.x + box2.w) - merged_x
        merged_h = max(box1.y + box1.h, box2.y + box2.h) - merged_y
        merged_box = Box(x=merged_x, y=merged_y, w=merged_w, h=merged_h)
        letter_boxes[i] = merged_box
        letter_boxes[j] = merged_box
    return letter_boxes


def _box_side_distance(box1: Box, box2: Box) -> float:
    """
    Calculate the distance between the closest two sides of two boxes.
    If the boxes overlap or touch, the distance is 0.
    """
    x1, y1, x2, y2 = box1.x, box1.y, box1.x + box1.w, box1.y + box1.h
    x3, y3, x4, y4 = box2.x, box2.y, box2.x + box2.w, box2.y + box2.h

    # Calculate horizontal and vertical distances
    horizontal_dist = 0.0
    vertical_dist = 0.0

    # If box1 is completely to the left of box2
    if x2 < x3:
        horizontal_dist = x3 - x2
    # If box1 is completely to the right of box2
    elif x4 < x1:
        horizontal_dist = x1 - x4

    # If box1 is completely above box2
    if y2 < y3:
        vertical_dist = y3 - y2
    # If box1 is completely below box2
    elif y4 < y1:
        vertical_dist = y1 - y4

    # If the boxes overlap or touch in both horizontal and vertical directions
    if horizontal_dist == 0 and vertical_dist == 0:
        return 0.0

    # Return the largest of the two distances in diagonal cases
    return max(horizontal_dist, vertical_dist)


def _mirror_around_horizontal_center(image: np.ndarray, boxes: list[LetterBox]) -> list[LetterBox]:
    """
    For some reason, the OCR flips the text boxes upside down.
    Mirror the boxes around the horizontal center of the image.
    """
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


def _find_word(word_cells: list[np.ndarray], language: str) -> str:
    words = [_parse_word_cell(cell, language=language) for cell in word_cells]
    log.info(f"Found words: {words}")
    if not words:
        return ""
    return max(words, key=len)  # type: ignore


def _parse_word_cell(cell: np.ndarray, language: str) -> str | None:
    config = "--psm 11"
    result = pytesseract.image_to_string(cell, lang=language, config=config)
    # TODO: Improve
    return result.strip()


def _parse_letter_boxes(result: str) -> list[LetterBox]:
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
