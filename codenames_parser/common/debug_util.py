import logging
import os
import time
from typing import Iterable, Sequence

import cv2
import numpy as np

from codenames_parser.common.models import P1P2, Box, Color, Line, Point


class Counter:
    def __init__(self):
        self.count = 0

    def add(self):
        self.count += 1

    def total(self):
        return self.count

    def __int__(self):
        return self.count

    def __str__(self):
        return str(self.count)

    def __format__(self, format_spec):
        return format(self.count, format_spec)


DEFAULT_RUN_ID = 9999999999 - int(time.time())
LINE_DRAW_SIZE = 1000
BOX_COLOR = (0, 255, 0)
HORIZONTAL_COLOR = np.array([20, 200, 20])
VERTICAL_COLOR = np.array([200, 20, 200])
RUN_COUNT = Counter()
SEPARATOR = "---------------------------------"
CONTEXT = ""

log = logging.getLogger(__name__)


def set_debug_context(context: str) -> None:
    global CONTEXT  # pylint: disable=global-statement
    CONTEXT = context


def save_debug_image(image: np.ndarray, title: str, show: bool = False) -> None:
    if not _is_debug_enabled():
        return
    run_folder = _get_run_folder()
    os.makedirs(run_folder, exist_ok=True)
    RUN_COUNT.add()
    file_name = f"{RUN_COUNT:03d}: {title}.jpg"
    file_path = os.path.join(run_folder, file_name)
    try:
        cv2.imwrite(file_path, image)
        if show:
            cv2.imshow(title, image)
    except Exception as e:
        log.debug(f"Error saving debug image: {e}")
        return


def draw_boxes(image: np.ndarray, boxes: Iterable[Box], title: str, thickness: int = 2):
    if not _is_debug_enabled():
        return
    image = image.copy()
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for box in boxes:
        top_left = (box.x, box.y)
        bottom_right = (box.x + box.w, box.y + box.h)
        cv2.rectangle(image, pt1=top_left, pt2=bottom_right, color=BOX_COLOR, thickness=thickness)
    save_debug_image(image, title=title)


def draw_lines(image: np.ndarray, lines: Iterable[Line], title: str):
    if not _is_debug_enabled():
        return
    # If image is grayscale, convert to BGR
    image = image.copy()
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for line in lines:
        loc = _get_line_draw_params(line)
        color = _pick_line_color(line)
        cv2.line(image, loc.p1, loc.p2, color, 2)
    save_debug_image(image, title=title)


def draw_polyline(image: np.ndarray, points: Sequence[np.ndarray], title: str):
    if not _is_debug_enabled():
        return
    image = image.copy()
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    pts = [np.array(points, dtype=np.int32)]
    cv2.polylines(image, pts=pts, isClosed=True, color=(0, 255, 0), thickness=2)
    save_debug_image(image, title=title)
    return


def _get_run_folder() -> str:
    debug_dir = os.getenv("DEBUG_OUTPUT_DIR", "debug")
    run_id = os.getenv("RUN_ID", str(DEFAULT_RUN_ID))
    run_dir = os.path.join(debug_dir, run_id)
    if CONTEXT:
        run_dir = os.path.join(run_dir, CONTEXT)
    return run_dir


def _get_line_draw_params(line: Line) -> P1P2:
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + LINE_DRAW_SIZE * (-b))
    x2 = int(x0 - LINE_DRAW_SIZE * (-b))
    y1 = int(y0 + LINE_DRAW_SIZE * (a))
    y2 = int(y0 - LINE_DRAW_SIZE * (a))
    p1 = Point(x1, y1)
    p2 = Point(x2, y2)
    return P1P2(p1, p2)


def _pick_line_color(line: Line) -> Color:
    color_1 = np.sin(line.theta)
    color_2 = 1 - color_1
    color: np.ndarray = color_1 * HORIZONTAL_COLOR + color_2 * VERTICAL_COLOR
    random_offset = np.random.randint(0, 50, 3)
    color += random_offset
    rounded = np.round(color)
    return Color(*rounded.tolist())


def _is_debug_enabled() -> bool:
    return os.getenv("DEBUG_DISABLED", "false").lower() not in ["true", "1"]
