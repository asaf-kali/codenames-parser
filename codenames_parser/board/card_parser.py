import logging

import numpy as np
import pytesseract

from codenames_parser.board.ocr import (
    TesseractResult,
    WordIndex,
    fetch_tesseract_language,
)
from codenames_parser.board.template_search import search_template
from codenames_parser.common.crop import crop_by_box
from codenames_parser.common.debug_util import (
    SEPARATOR,
    draw_boxes,
    save_debug_image,
    set_debug_context,
)
from codenames_parser.common.general import quantize, sharpen
from codenames_parser.common.image_reader import read_image
from codenames_parser.common.models import Box
from codenames_parser.common.scale import resize_image
from codenames_parser.resources.resource_manager import get_card_template_path

log = logging.getLogger(__name__)


def parse_cards(cells: list[np.ndarray], language: str) -> list[str]:
    cards = []
    card_template = read_image(get_card_template_path())
    for i, cell in enumerate(cells):
        set_debug_context(f"card {i}")
        log.info(f"\n{SEPARATOR}")
        log.info(f"Processing card {i}")
        card = _parse_card(i=i, image=cell, language=language, card_template=card_template)
        cards.append(card)
    return cards


def _parse_card(i: int, image: np.ndarray, language: str, card_template: np.ndarray) -> str:
    save_debug_image(image, title="original card")
    actual_card = search_template(source=image, template=card_template)
    save_debug_image(actual_card, title=f"copped card {i}", important=True)
    text_section = _text_section_crop(actual_card)
    text_section_processed = _process_text_section(text_section)
    text = _extract_text(text_section_processed, language=language)
    save_debug_image(text_section, title=f"text section: {text}", important=True)
    return text


def _text_section_crop(card: np.ndarray) -> np.ndarray:
    # Example:
    # size = 500x324
    # top_left = (50, 190)
    # w, h = (400, 90)
    log.debug("Cropping text section...")
    width, height = card.shape[1], card.shape[0]
    text_x = int(width * 0.1)
    text_y = int(height * 0.55)
    text_width = int(width * 0.8)
    text_height = int(height * 0.32)
    box = Box(x=text_x, y=text_y, w=text_width, h=text_height)
    text_section = crop_by_box(card, box=box)
    save_debug_image(text_section, title="text section")
    return text_section


def _process_text_section(text_section: np.ndarray) -> np.ndarray:
    resized = resize_image(image=text_section, dst_width=500)
    sharpened = sharpen(image=resized)
    quantized = quantize(image=sharpened, k=6)
    save_debug_image(quantized, title="text section for parsing", important=True)
    return quantized


def _extract_text(card: np.ndarray, language: str) -> str:
    log.debug("Extracting text...")
    fetch_tesseract_language(language)
    config = "--psm 11"
    data = pytesseract.image_to_data(card, output_type=pytesseract.Output.DICT, lang=language, config=config)
    results = parse_tesseract_data(data)
    boxes = [result.box for result in results]
    draw_boxes(image=card, boxes=boxes, title="tesseract boxes", thickness=3)
    word = _pick_word_from_results(results)
    log.info(f"Extracted word: [{word}]")
    return word.strip()


def _pick_word_from_results(results: list[TesseractResult]) -> str:
    words = [result.text for result in results]
    log.info(f"Extracted words: {words}")
    for result in results:
        result.text = _keep_only_letters(result.text)
    good_results = [result for result in results if len(result.text) > 1 and result.confidence > 30]
    if not good_results:
        log.warning("No good results extracted")
        return ""
    good_results.sort(key=lambda result: result.confidence, reverse=True)
    if len(good_results) > 1:
        good_words = [result.text for result in good_results]
        log.info(f"Good words: {good_words}")
    best_result = good_results[0]
    log.info(f"Best result: [{best_result.text}]")
    return best_result.text


# def _pick_word_from_raw_text(raw_text: str) -> str:
#     words = raw_text.split()
#     log.debug(f"Extracted words raw: {words}")
#     words_alphabet = [_keep_only_letters(word) for word in words]
#     words_filtered = [word for word in words_alphabet if len(word) > 1]
#     words_sorted = sorted(words_filtered, key=len, reverse=True)
#     log.debug(f"Sorted words: {words_sorted}")
#     if not words_sorted:
#         log.warning("No words extracted")
#         return ""
#     longest_word = words_sorted[0]
#     return longest_word


def _keep_only_letters(text: str) -> str:
    return "".join(filter(str.isalpha, text))  # type: ignore


def parse_tesseract_data(data: dict) -> list[TesseractResult]:
    results = []
    n_boxes = len(data["text"])
    for i in range(n_boxes):
        result = _parse_result(data, i)
        results.append(result)
    return results


def _parse_result(data: dict, i: int) -> TesseractResult:
    text = data["text"][i]
    confidence = int(data["conf"][i])
    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
    box = Box(x=x, y=y, w=w, h=h)
    level = int(data["level"][i])
    index = WordIndex(
        page=int(data["page_num"][i]),
        block=int(data["block_num"][i]),
        paragraph=int(data["par_num"][i]),
        line=int(data["line_num"][i]),
    )
    result = TesseractResult(text=text, confidence=confidence, box=box, level=level, index=index)
    return result
