import logging
import os
import time

import cv2
import numpy as np

from codenames_parser.models import P1P2, Color, Line, Point

DEFAULT_RUN_ID = 9999999999 - int(time.time())
run_count = 0
SEPARATOR = "---------------------------------"
log = logging.getLogger(__name__)


def save_debug_image(image: np.ndarray, title: str, show: bool = False) -> None:
    debug_disabled = os.getenv("DEBUG_DISABLED", "false").lower() in ["true", "1"]
    if debug_disabled:
        return
    global run_count
    debug_dir = os.getenv("DEBUG_OUTPUT_DIR", "debug")
    run_id = os.getenv("RUN_ID", str(DEFAULT_RUN_ID))
    run_folder = os.path.join(debug_dir, run_id)
    os.makedirs(run_folder, exist_ok=True)
    run_count += 1
    file_name = f"{run_count:03d}: {title}.jpg"
    file_path = os.path.join(run_folder, file_name)
    try:
        cv2.imwrite(file_path, image)
        if show:
            cv2.imshow(title, image)
    except Exception as e:
        log.debug(f"Error saving debug image: {e}")
        return


def draw_lines(image: np.ndarray, lines: list[Line], title: str) -> np.ndarray:
    # If image is grayscale, convert to BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for line in lines:
        loc = _get_line_draw_params(line)
        color = _pick_line_color(line)
        cv2.line(image, loc.p1, loc.p2, color, 2)
    save_debug_image(image, title=title)
    return image


def _get_line_draw_params(line: Line) -> P1P2:
    SIZE = 1000
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + SIZE * (-b))
    x2 = int(x0 - SIZE * (-b))
    y1 = int(y0 + SIZE * (a))
    y2 = int(y0 - SIZE * (a))
    p1 = Point(x1, y1)
    p2 = Point(x2, y2)
    return P1P2(p1, p2)


def _pick_line_color(line: Line) -> Color:
    red = np.sin(line.theta)
    blue = 1 - red
    color = 200 * np.array([blue, 0, red])
    random_offset = np.random.randint(0, 50, 3)
    color += random_offset
    rounded = np.round(color)
    return Color(*rounded.tolist())
