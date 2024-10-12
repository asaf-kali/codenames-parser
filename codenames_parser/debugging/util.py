import logging
import os
import time

import cv2
import numpy as np

DEFAULT_RUN_ID = 9999999999 - int(time.time())
run_count = 0

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
