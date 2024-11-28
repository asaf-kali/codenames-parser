import logging
from dataclasses import dataclass

import numpy as np
import pytesseract

from codenames_parser.board.ocr import (
    TesseractResult,
    fetch_tesseract_language,
    parse_tesseract_data,
)
from codenames_parser.board.template_search import search_template
from codenames_parser.common.impr.color_manipulation import quantize
from codenames_parser.common.impr.crop import crop_by_box
from codenames_parser.common.impr.general import sharpen
from codenames_parser.common.impr.reader import read_image
from codenames_parser.common.impr.scale import resize_image
from codenames_parser.common.resources.resource_manager import get_card_template_path
from codenames_parser.common.utils.debug_util import (
    SEPARATOR,
    draw_boxes,
    save_debug_image,
    set_debug_context,
)
from codenames_parser.common.utils.models import Box, Size

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
    save_debug_image(actual_card, title=f"copped card {i}")
    text_section = _text_section_crop(actual_card)
    text_section_processed = _process_text_section(text_section)
    text = _extract_text(text_section_processed, language=language)
    # save_debug_image(text_section, title=f"text section: {text}", important=True)
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


def _process_text_section(text_section: np.ndarray, quantization_k: int = 6) -> np.ndarray:
    resized = resize_image(image=text_section, dst_width=500)
    sharpened = sharpen(image=resized)
    quantized = quantize(image=sharpened, k=quantization_k)
    save_debug_image(quantized, title="text section for parsing", important=True)
    return quantized


def _extract_text(image: np.ndarray, language: str) -> str:
    fetch_tesseract_language(language)
    config = "--psm 11"
    log.info("Running OCR...")
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang=language, config=config)
    results = parse_tesseract_data(data)
    boxes = [result.box for result in results]
    draw_boxes(image=image, boxes=boxes, title="tesseract boxes", thickness=3)
    word = _pick_word_from_results(image=image, results=results)
    log.info(f"Extracted word: [{word}]")
    return word.strip()


def _pick_word_from_results(image: np.ndarray, results: list[TesseractResult]) -> str:
    word_results = [result for result in results if result.is_word]
    words = [result.text for result in word_results]
    log.info(f"Extracted words: {words}")
    for result in word_results:
        result.text = _keep_only_letters(result.text)
    possible_results = [result for result in word_results if len(result.text) > 1]
    if not possible_results:
        log.warning("No good results extracted")
        return ""
    if len(possible_results) > 1:
        possible_words = [result.text for result in possible_results]
        log.info(f"Good words: {possible_words}")
        grading_context = _create_grading_context(image)
        grades = {result.text: _grade_result(result, grading_context=grading_context) for result in possible_results}
        grades_sorted = dict(sorted(grades.items(), key=lambda item: item[1], reverse=True))
        log.info(f"Words grades: {grades_sorted}")
        possible_results.sort(key=lambda result: grades[result.text], reverse=True)
    best_result = possible_results[0]
    return best_result.text


def _keep_only_letters(text: str) -> str:
    return "".join(filter(str.isalpha, text))  # type: ignore


@dataclass
class GradingContext:
    image_size: Size
    expected_letter_width: float
    expected_letter_height: float


def _create_grading_context(image: np.ndarray) -> GradingContext:
    size = Size(width=image.shape[1], height=image.shape[0])
    expected_letter_width = size.width / 20  # We expect around 20 letters to fit in the text section
    expected_letter_height = size.height / 1.5  # We expect the word to be around half of the text section height
    return GradingContext(
        image_size=size,
        expected_letter_width=expected_letter_width,
        expected_letter_height=expected_letter_height,
    )


def _grade_result(result: TesseractResult, grading_context: GradingContext) -> float:
    word_length = len(result.text)
    expected_width = grading_context.expected_letter_width * word_length
    expected_height = grading_context.expected_letter_height
    actual_width, actual_height = result.box.w, result.box.h
    width_distance = _distance_from_range(actual_width, expected_width * 0.8, expected_width * 1.2)
    height_distance = _distance_from_range(actual_height, expected_height * 0.8, expected_height * 1.2)
    width_distance += 1
    height_distance += 1
    expected_area = expected_width * expected_height
    actual_area = actual_width * actual_height
    grade = round(1000 / (width_distance * height_distance), 3)
    log.info(
        f"Word: [{result.text}], Length: [{word_length}], "
        f"Expected area: [{expected_area:.0f}], Actual area: [{actual_area:.0f}]"
    )
    return grade


def _distance_from_range(number: float, lower: float, upper: float) -> float:
    if lower <= number <= upper:
        return 0
    limit = lower if number < lower else upper
    return abs(number - limit)


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
