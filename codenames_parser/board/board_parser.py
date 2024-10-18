# pylint: disable=R0801

import numpy as np
from codenames.game.board import Board

from codenames_parser.board.card_parser import parse_cards
from codenames_parser.board.grid_detection import extract_boxes
from codenames_parser.common.align import align_image, apply_rotations
from codenames_parser.common.crop import crop_by_box
from codenames_parser.common.debug_util import draw_boxes
from codenames_parser.common.image_reader import read_image
from codenames_parser.common.models import Box
from codenames_parser.common.scale import scale_down_image


def parse_board(image_path: str, language: str) -> Board:
    image = read_image(image_path)
    scale_result = scale_down_image(image)
    alignment_result = align_image(scale_result.image)
    boxes = extract_boxes(image=alignment_result.aligned_image)
    cells = _crop_cells(
        image=image,
        boxes=boxes,
        scale_factor=1 / scale_result.scale_factor,
        rotations=alignment_result.rotations,
        enlarge_factor=0.20,
    )
    cards = parse_cards(cells, language=language)
    return Board(cards=cards, language=language)


def _crop_cells(
    image: np.ndarray, boxes: list[Box], scale_factor: float, rotations: list[float], enlarge_factor: float
) -> list[np.ndarray]:
    image_rotated = apply_rotations(image, rotations=rotations)
    boxes_scaled = [_scale_box(box, scale_factor) for box in boxes]
    draw_boxes(image_rotated, boxes=boxes_scaled, title="boxes scaled")
    boxes_enlarged = [_box_enlarged(box, factor=enlarge_factor) for box in boxes_scaled]
    draw_boxes(image_rotated, boxes=boxes_enlarged, title="boxes enlarged")
    cells = [crop_by_box(image_rotated, box=box) for box in boxes_enlarged]
    return cells


def _scale_box(box: Box, scale_factor: float) -> Box:
    return Box(
        x=int(box.x * scale_factor),
        y=int(box.y * scale_factor),
        w=int(box.w * scale_factor),
        h=int(box.h * scale_factor),
    )


def _box_enlarged(box: Box, factor: float) -> Box:
    x_diff = box.w * factor
    y_diff = box.h * factor
    return Box(
        x=int(box.x - x_diff / 2),
        y=int(box.y - x_diff / 2),
        w=int(box.w + x_diff),
        h=int(box.h + y_diff),
    )


# def _translate_boxes(boxes: list[Box], scale_factor: float, rotation_degrees: float) -> list[Box]:
#     translation_matrix = cv2.getRotationMatrix2D(center=(0, 0), angle=rotation_degrees, scale=1 / scale_factor)
#
#     def translate_box(box: Box) -> Box:
#         x1, y1 = box.x, box.y
#         x2, y2 = box.x + box.w, box.y + box.h
#         top_left = (x1, y1, 1)
#         bottom_right = (x2, y2, 1)
#         new_top_left = translation_matrix @ top_left
#         new_bottom_right = translation_matrix @ bottom_right
#         return Box(
#             x=int(new_top_left[0]),
#             y=int(new_top_left[1]),
#             w=int(new_bottom_right[0] - new_top_left[0]),
#             h=int(new_bottom_right[1] - new_top_left[1]),
#         )
#
#     translated_boxes = [translate_box(box) for box in boxes]
#     return translated_boxes
