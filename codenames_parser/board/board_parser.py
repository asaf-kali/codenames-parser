# pylint: disable=R0801

import numpy as np

from codenames_parser.board.card_parser import parse_cards
from codenames_parser.board.grid_detection import extract_boxes
from codenames_parser.common.impr.align import align_image, apply_rotations
from codenames_parser.common.impr.crop import crop_by_box
from codenames_parser.common.impr.scale import scale_down_image
from codenames_parser.common.utils.debug_util import draw_boxes
from codenames_parser.common.utils.models import Box


def parse_board(image: np.ndarray, language: str) -> list[str]:
    scale_result = scale_down_image(image)
    alignment_result = align_image(scale_result.image)
    boxes = extract_boxes(image=alignment_result.aligned_image)
    cells = _crop_cells(
        image=image,
        boxes=boxes,
        scale_factor=1 / scale_result.scale_factor,
        rotations=alignment_result.rotations,
        enlarge_factor=0.25,
    )
    cards = parse_cards(cells, language=language)
    return cards


def _crop_cells(
    image: np.ndarray, boxes: list[Box], scale_factor: float, rotations: list[float], enlarge_factor: float
) -> list[np.ndarray]:
    image_rotated = apply_rotations(image, rotations=rotations)
    boxes_scaled = [_scale_box(box, scale_factor) for box in boxes]
    draw_boxes(image_rotated, boxes=boxes_scaled, title="boxes scaled", thickness=5)
    boxes_enlarged = [_box_enlarged(box, factor=enlarge_factor) for box in boxes_scaled]
    draw_boxes(image_rotated, boxes=boxes_enlarged, title="boxes enlarged", thickness=5)
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
