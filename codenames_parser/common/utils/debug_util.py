import logging
import os
import shutil
import time
from collections import defaultdict
from functools import lru_cache
from typing import Iterable, Sequence

import cv2
import numpy as np

from codenames_parser.common.utils.models import P1P2, Box, Color, Line, Point

# Environment variables
DEBUG_ENV_VAR = "SAVE_DEBUG_IMAGES"
OUTPUT_DIR_ENV_VAR = "DEBUG_OUTPUT_DIR"
RUN_ID_ENV_VAR = "RUN_ID"

# Constants
DEFAULT_RUN_ID = 9999999999 - int(time.time())
LINE_DRAW_SIZE = 1000
BOX_COLOR = (0, 255, 0)
HORIZONTAL_COLOR = np.array([20, 200, 20])
VERTICAL_COLOR = np.array([200, 20, 200])
SEPARATOR = "---------------------------------"
CONTEXT = ""

log = logging.getLogger(__name__)


def set_save_debug_images(enabled: bool) -> None:
    os.environ[DEBUG_ENV_VAR] = str(enabled).lower()


def set_debug_output_dir(output_dir: str) -> None:
    os.environ[OUTPUT_DIR_ENV_VAR] = output_dir


def set_run_id(run_id: str) -> None:
    os.environ[RUN_ID_ENV_VAR] = run_id


def set_debug_context(context: str) -> None:
    global CONTEXT  # pylint: disable=global-statement
    CONTEXT = context


def save_debug_image(image: np.ndarray, title: str, show: bool = False, important: bool = False) -> str | None:
    if not _is_debug_enabled():
        return None
    file_path = get_debug_file_path(title)
    try:
        cv2.imwrite(file_path, image)
        if show:
            cv2.imshow(title, image)
        if important:
            important_path = get_debug_file_path(title, important=True)
            shutil.copy(file_path, important_path)
    except Exception as e:
        log.debug(f"Error saving debug image: {e}")
        return None
    return file_path


def save_plt_image(plt, title: str, show: bool = False) -> str | None:
    if not _is_debug_enabled():
        return None
    file_path = get_debug_file_path(title)
    try:
        plt.savefig(file_path)
        if show:
            plt.show()
    except Exception as e:
        log.debug(f"Error saving debug image: {e}")
        return None
    return file_path


def get_debug_file_path(title: str, important: bool = False) -> str:
    run_folder = _get_folder(important=important)
    counter = _get_counter()
    counter[run_folder] += 1
    os.makedirs(run_folder, exist_ok=True)
    file_name = f"{counter[run_folder]:03d}: {title}.jpg"
    file_path = os.path.join(run_folder, file_name)
    return file_path


@lru_cache
def _get_counter() -> dict:
    return defaultdict(int)


def draw_boxes(image: np.ndarray, boxes: Iterable[Box], title: str, thickness: int = 2) -> str | None:
    if not _is_debug_enabled():
        return None
    image = image.copy()
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for box in boxes:
        top_left = (box.x, box.y)
        bottom_right = (box.x + box.w, box.y + box.h)
        cv2.rectangle(image, pt1=top_left, pt2=bottom_right, color=BOX_COLOR, thickness=thickness)
    return save_debug_image(image, title=title)


def draw_lines(image: np.ndarray, lines: Iterable[Line], title: str) -> str | None:
    if not _is_debug_enabled():
        return None
    # If image is grayscale, convert to BGR
    image = image.copy()
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for line in lines:
        loc = _get_line_draw_params(line)
        color = _pick_line_color(line)
        cv2.line(image, loc.p1, loc.p2, color, 2)
    return save_debug_image(image, title=title)


def draw_polyline(image: np.ndarray, points: Sequence[np.ndarray], title: str, important: bool = False) -> str | None:
    if not _is_debug_enabled():
        return None
    image = image.copy()
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    pts = [np.array(points, dtype=np.int32)]
    cv2.polylines(image, pts=pts, isClosed=True, color=(0, 255, 0), thickness=2)
    return save_debug_image(image, title=title, important=important)


def draw_points(image: np.ndarray, points: Sequence[Point], title: str, radius: int = 3) -> str | None:
    if not _is_debug_enabled():
        return None
    image = image.copy()
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for point in points:
        cv2.circle(image, point, radius, (0, 255, 0), -1)
    return save_debug_image(image, title=title)


def _get_folder(important: bool) -> str:
    debug_dir = os.getenv(OUTPUT_DIR_ENV_VAR, "debug")
    run_id = os.getenv(RUN_ID_ENV_VAR, str(DEFAULT_RUN_ID))
    run_dir = os.path.join(debug_dir, run_id)
    if important:
        run_dir = os.path.join(run_dir, "\x20important")
    elif CONTEXT:
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
    return os.getenv(DEBUG_ENV_VAR, "true").lower() in ["true", "1"]
