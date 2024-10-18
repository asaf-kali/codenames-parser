import logging
from typing import Tuple

import cv2
import numpy as np

from codenames_parser.common.debug_util import save_debug_image

log = logging.getLogger(__name__)


def downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    """Downsample the image by the given factor.

    Args:
        image (np.ndarray): Input image.
        factor (int): Downsampling factor.

    Returns:
        np.ndarray: Downsampled image.
    """
    if factor == 1:
        return image
    height, width = image.shape[:2]
    new_size = (width // factor, height // factor)
    downsampled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return downsampled_image


def transform_template(template: np.ndarray, angle: float, scale: float, factor: int) -> np.ndarray:
    """Rotate and scale the template image, including downsampling factor.

    Args:
        template (np.ndarray): Template image.
        angle (float): Rotation angle in degrees.
        scale (float): Scaling factor.
        factor (int): Downsampling factor.
    Returns:
        np.ndarray: Transformed template image.
    """
    # Compute the overall scaling factor
    overall_scale = scale / factor
    # Resize the template
    height, width = template.shape[:2]
    new_width = int(width * overall_scale)
    new_height = int(height * overall_scale)
    resized_template = cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    # Rotate the resized template
    center = (new_width / 2, new_height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # Already scaled
    rotated_template = cv2.warpAffine(
        resized_template,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated_template


def compute_psr(match_result: np.ndarray, peak_coords: tuple[int, int]) -> float:
    """Compute the Peak-to-Sidelobe Ratio (PSR).

    Args:
        match_result (np.ndarray): Result from template matching.
        peak_coords (Tuple[int, int]): Coordinates of the peak correlation value.

    Returns:
        float: PSR value.
    """
    if peak_coords == (0, 0):
        return 0.0
    peak_value = match_result[peak_coords[1], peak_coords[0]]

    # Exclude a small region around the peak
    mask = np.ones_like(match_result, dtype=bool)
    h, w = match_result.shape
    peak_x, peak_y = peak_coords
    exclude_size = int(min(h, w) * 0.1)  # Exclude 10% of the smallest dimension
    x_start = max(0, peak_x - exclude_size)
    x_end = min(w, peak_x + exclude_size + 1)
    y_start = max(0, peak_y - exclude_size)
    y_end = min(h, peak_y + exclude_size + 1)
    mask[y_start:y_end, x_start:x_end] = False

    sidelobe = match_result[mask]
    mean_sidelobe = np.mean(sidelobe)
    std_sidelobe = np.std(sidelobe)

    # Avoid division by zero
    if std_sidelobe == 0:
        return 0.0

    psr = (peak_value - mean_sidelobe) / std_sidelobe
    return float(psr)


def find_best_match(match_result: np.ndarray) -> Tuple[int, int, float]:
    """Find the best match location and compute its PSR value.

    Args:
        match_result (np.ndarray): Result from template matching.

    Returns:
        Tuple[int, int, float]: Best location (x, y) and PSR value.
    """
    _, _, _, max_loc = cv2.minMaxLoc(match_result)
    psr_value = compute_psr(match_result, max_loc)  # type: ignore[arg-type]
    return max_loc[0], max_loc[1], psr_value


def match_template(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Perform template matching using normalized cross-correlation.

    Args:
        source (np.ndarray): Source image.
        template (np.ndarray): Template image.

    Returns:
        np.ndarray: Matching result.
    """
    result = cv2.matchTemplate(source, template, method=cv2.TM_CCOEFF_NORMED)
    # result_image = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # type: ignore[call-overload]
    # save_debug_image(result_image, title="match result")
    return result


def _ensure_graysacle(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# pylint: disable=too-many-statements
def pyramid_image_search(source_image: np.ndarray, template_image: np.ndarray, num_iterations: int = 3) -> np.ndarray:
    """Search for the template location in the source image using pyramid search.

    Args:
        source_image (np.ndarray): Source image.
        template_image (np.ndarray): Template image.
        num_iterations (int, optional): Number of iterations. Defaults to 5.

    Returns:
        np.ndarray: Matched region from the source image.
    """
    source_gray = _ensure_graysacle(source_image)
    template_gray = _ensure_graysacle(template_image)
    # Initial angle and scale ranges
    scale_ratio = max(template_image.shape[0] / source_image.shape[0], template_image.shape[1] / source_image.shape[1])
    min_scale, max_scale = 0.03, round(1.0 / scale_ratio, 3)
    min_angle, max_angle = (-30, 30)

    # Initial step sizes
    angle_step_num = 5
    scale_step_num = 3
    angle_step = 5.0
    scale_step = 0.1

    # Initial best values
    best_angle = 0.0
    best_scale = max_scale
    best_location = (0, 0)
    best_template = iteration_best_template = template_gray
    # best_value = -np.inf

    # Iterate
    for i in range(1, num_iterations + 1):
        # Downsample factor
        factor = 2 ** (num_iterations - i)
        log.info(f"Iteration {i}: downsample factor={factor}")
        source_downsampled = downsample_image(source_gray, factor)
        save_debug_image(source_downsampled, title=f"source downsampled {i}")
        # template_downsampled = scale_down_image(image=template_gray, max_dimension=max(source_downsampled.shape)).image  # noqa pylint: disable=line-too-long
        # Update angle and scale ranges
        iter_angle_min = max(min_angle, best_angle - angle_step * angle_step_num)
        iter_angle_max = min(max_angle, best_angle + angle_step * angle_step_num)
        iter_angles = np.linspace(iter_angle_min, iter_angle_max, num=angle_step_num * 2 + 1)

        iter_scale_min = max(min_scale, best_scale - scale_step * scale_step_num)
        iter_scale_max = min(max_scale, best_scale + scale_step * scale_step_num)
        iter_scales = np.linspace(iter_scale_min, iter_scale_max, num=scale_step_num * 2 + 1)

        # Variables to store best match in this iteration
        iteration_best_value = -np.inf
        iteration_best_angle = best_angle
        iteration_best_scale = best_scale
        iteration_best_location = best_location

        # For each angle and scale
        for angle in iter_angles:
            for scale in iter_scales:
                if scale > max_scale:
                    continue
                # Transform template
                template_transformed = transform_template(template_gray, angle, scale, factor=factor)
                # save_debug_image(template_transformed, title=f"template transformed {i} ({angle:.2f}°, X{scale:.2f})")
                # Perform template matching
                match_result = match_template(source_downsampled, template_transformed)
                # Find best match
                max_loc_x, max_loc_y, max_val = find_best_match(match_result)
                # Update best match if necessary
                if max_val > iteration_best_value:
                    iteration_best_value = max_val
                    iteration_best_angle = angle
                    iteration_best_scale = scale
                    iteration_best_location = (max_loc_x, max_loc_y)
                    iteration_best_template = template_transformed.copy()

        # Update best values for next iteration
        best_angle = iteration_best_angle
        best_scale = iteration_best_scale
        best_location = iteration_best_location
        best_value = iteration_best_value
        best_template = iteration_best_template
        save_debug_image(best_template, title=f"best template {i} ({best_angle:.2f}°, X{best_scale:.2f})")
        log.info(f"Iteration {i}: angle={best_angle:<6.2f} scale={best_scale:<6.2f} value={best_value:<6.2f}")

        # Narrow down the angle and scale steps
        angle_step = angle_step / 2.0
        scale_step = scale_step / 2.0
        log.debug(f"New step sizes: angle={angle_step:<6.2f} scale={scale_step:<6.2f}")

    # Extract the matched region
    top_left_x = best_location[0]
    top_left_y = best_location[1]
    template_h, template_w = best_template.shape[:2]
    bottom_right_x = top_left_x + template_w
    bottom_right_y = top_left_y + template_h

    # Ensure indices are within image boundaries
    height, width = source_image.shape[:2]
    top_left_x = max(0, min(top_left_x, width - 1))
    top_left_y = max(0, min(top_left_y, height - 1))
    bottom_right_x = max(0, min(bottom_right_x, width))
    bottom_right_y = max(0, min(bottom_right_y, height))

    matched_image = source_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    save_debug_image(matched_image, title="matched image")
    return matched_image
