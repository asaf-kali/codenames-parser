# pylint: disable=R0801

import cv2
from codenames.game.board import Board

from codenames_parser.board.card_parser import parse_cards
from codenames_parser.board.grid_detection import extract_boxes
from codenames_parser.common.align import align_image
from codenames_parser.common.grid_detection import crop_cells
from codenames_parser.common.image_reader import read_image
from codenames_parser.common.models import Box
from codenames_parser.common.scale import scale_down_image


def parse_board(image_path: str, language: str) -> Board:
    image = read_image(image_path)
    scale_result = scale_down_image(image)
    alignment_result = align_image(scale_result.image)
    boxes = extract_boxes(alignment_result.aligned_image)
    boxes_scaled = _translate_boxes(
        boxes=boxes,
        scale_factor=scale_result.scale_factor,
        rotation_degrees=alignment_result.rotation_degrees,
    )
    cells = crop_cells(image=image, boxes=boxes_scaled)
    cards = parse_cards(cells, language=language)
    return Board(cards=cards, language=language)


def _translate_boxes(boxes: list[Box], scale_factor: float, rotation_degrees: float) -> list[Box]:
    translation_matrix = cv2.getRotationMatrix2D(center=(0, 0), angle=rotation_degrees, scale=1 / scale_factor)

    def translate_box(box: Box) -> Box:
        x1, y1 = box.x, box.y
        x2, y2 = box.x + box.w, box.y + box.h
        top_left = (x1, y1, 1)
        bottom_right = (x2, y2, 1)
        new_top_left = translation_matrix @ top_left
        new_bottom_right = translation_matrix @ bottom_right
        return Box(
            x=int(new_top_left[0]),
            y=int(new_top_left[1]),
            w=int(new_bottom_right[0] - new_top_left[0]),
            h=int(new_bottom_right[1] - new_top_left[1]),
        )

    translated_boxes = [translate_box(box) for box in boxes]
    return translated_boxes
