import logging

from pytesseract import pytesseract

from codenames_parser.board.card_parser import _pick_word_from_raw_text
from codenames_parser.board.ocr import fetch_tesseract_language
from codenames_parser.common.image_reader import read_image

image = read_image("data/board1_top.jpg")


logging.basicConfig(level=logging.DEBUG)

fetch_tesseract_language("heb")
config = "--psm 11"
result = pytesseract.image_to_string(image, lang="heb", config=config)
word = _pick_word_from_raw_text(result)
print(word)
