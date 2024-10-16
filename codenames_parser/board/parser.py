# pylint: disable=R0801
from codenames.game.card import Card

from codenames_parser.board.card_parser import parse_cards
from codenames_parser.board.grid_detection import extract_cells
from codenames_parser.common.align import align_image
from codenames_parser.common.crop import crop_image
from codenames_parser.common.image_reader import read_image
from codenames_parser.common.models import Grid
from codenames_parser.common.scale import scale_down_image


def parse_board(image_path: str) -> Grid[Card]:
    image = read_image(image_path)
    small_image = scale_down_image(image)
    aligned_image = align_image(small_image)
    cropped = crop_image(aligned_image)
    cells = extract_cells(cropped)
    cards = parse_cards(cells)
    return cards
