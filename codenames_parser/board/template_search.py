import logging

import cv2
import numpy as np

from codenames_parser.common.align import apply_rotation
from codenames_parser.common.debug_util import save_debug_image
from codenames_parser.common.models import Point

log = logging.getLogger(__name__)


# pylint: disable=too-many-statements
def pyramid_image_search(source_image: np.ndarray, template_image: np.ndarray, num_iterations: int = 2) -> np.ndarray:
    """Search for the template location in the source image using pyramid search.

    Args:
        source_image (np.ndarray): Source image.
        template_image (np.ndarray): Template image.
        num_iterations (int, optional): Number of iterations. Defaults to 5.

    Returns:
        np.ndarray: Matched region from the source image.
    """
    # Convert to grayscale if necessary
    source_gray = _ensure_graysacle(source_image)
    template_gray = _ensure_graysacle(template_image)
    # Initial angle and scale ranges
    scale_ratio = max(template_image.shape[0] / source_image.shape[0], template_image.shape[1] / source_image.shape[1])
    min_angle, max_angle = (-5, 5)
    min_scale, max_scale = 0.1, round(1.0 / scale_ratio, 4)
    angle_step_num = 6
    scale_step_num = 4
    iter_angles = np.linspace(min_angle, max_angle, num=angle_step_num * 2 + 1)
    iter_scales = np.linspace(min_scale, max_scale, num=scale_step_num * 2 + 1)

    # Initial best values
    best_angle = 0.0
    best_scale = max_scale
    best_location = Point(0, 0)
    best_template = iteration_best_template = template_gray

    # Iterate
    for i in range(1, num_iterations + 1):
        # Downsample factor
        factor = 2 ** (num_iterations - i)
        log.info(f"Iteration {i}: downsample factor={factor}")
        source_gray = apply_rotation(image=source_gray, angle_degrees=-best_angle)
        source_downsample = downsample_image(source_gray, factor=factor)
        save_debug_image(source_downsample, title=f"source downsample {i}")
        # template_downsampled = scale_down_image(image=template_gray, max_dimension=max(source_downsampled.shape)).image  # noqa pylint: disable=line-too-long

        # Update angle and scale ranges
        # iter_angle_min = max(min_angle, best_angle - angle_step * angle_step_num)
        # iter_angle_max = min(max_angle, best_angle + angle_step * angle_step_num)
        # iter_angles = np.linspace(iter_angle_min, iter_angle_max, num=angle_step_num * 2 + 1)
        # iter_scale_min = max(min_scale, best_scale - scale_step * scale_step_num)
        # iter_scale_max = min(max_scale, best_scale + scale_step * scale_step_num)
        # iter_scales = np.linspace(iter_scale_min, iter_scale_max, num=scale_step_num * 2 + 1)
        # log.info(f"Angle range: {iter_angle_min < 5:.2f} to {iter_angle_max:.2f}")
        # log.info(f"Scale range: {iter_scale_min < 5:.2f} to {iter_scale_max:.2f}")

        # Variables to store best match in this iteration
        iteration_best_value = -np.inf
        iteration_best_angle = 0.0
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
                match_result = match_template(source_downsample, template_transformed)
                # Find best match
                max_loc_x, max_loc_y, max_val = find_best_match(match_result)
                # Update best match if necessary
                if max_val > iteration_best_value:
                    iteration_best_value = max_val
                    iteration_best_angle = angle
                    iteration_best_scale = scale
                    iteration_best_location = Point(max_loc_x, max_loc_y)
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
        # angle_step = angle_step / 2.0
        # scale_step = scale_step / 2.0
        # log.debug(f"New step sizes: angle={angle_step:<6.2f} scale={scale_step:<6.2f}")

    matched_image = _crop_best_result(
        source_gray,
        best_angle=best_angle,
        best_location=best_location,
        best_template=best_template,
    )
    return matched_image


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
        peak_coords (tuple[int, int]): Coordinates of the peak correlation value.

    Returns:
        float: PSR value.
    """
    if peak_coords == (0, 0):
        return -np.inf
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
        return -np.inf

    psr = (peak_value - mean_sidelobe) / std_sidelobe
    return float(psr)


def find_best_match(match_result: np.ndarray) -> tuple[int, int, float]:
    """Find the best match location and compute its PSR value.

    Args:
        match_result (np.ndarray): Result from template matching.

    Returns:
        tuple[int, int, float]: Best location (x, y) and PSR value.
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


def _crop_best_result(
    image: np.ndarray,
    best_angle: float,
    best_location: Point,
    best_template: np.ndarray,
) -> np.ndarray:
    """Crop the matched region from the source image, taking rotation into account.

    Args:
        image (np.ndarray): Original source image.
        best_angle (float): Best rotation angle found.
        best_location (tuple[int, int]): Top-left corner location of the match in the source image.
        best_template (np.ndarray): Rotated template used for the best match.

    Returns:
        np.ndarray: Cropped matched region from the source image.
    """
    image_rotated = apply_rotation(image=image, angle_degrees=-best_angle)
    save_debug_image(image_rotated, title="rotated image")
    # Get the size of the rotated template
    template_h, template_w = best_template.shape[:2]

    # Coordinates of the top-left corner in the source image
    top_left_x = best_location[0]
    top_left_y = best_location[1]

    # Center of the template in its own coordinate system
    # template_center = np.array([template_w / 2, template_h / 2])

    # Define the corners of the rotated template relative to its center
    corners = np.array(
        [
            [-template_w / 2, -template_h / 2],
            [template_w / 2, -template_h / 2],
            [template_w / 2, template_h / 2],
            [-template_w / 2, template_h / 2],
        ]
    )

    # Rotation matrix
    angle_rad = np.deg2rad(0)
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )

    # Rotate corners
    rotated_corners = np.dot(corners, rotation_matrix.T)

    # Shift corners to the match location in the source image
    matched_corners = rotated_corners + np.array([top_left_x + template_w / 2, top_left_y + template_h / 2])

    # Get bounding rectangle of the rotated template
    x_coords = matched_corners[:, 0]
    y_coords = matched_corners[:, 1]

    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)

    # Create a mask for the rotated template
    mask = np.zeros_like(image, dtype=np.uint8)  # type: ignore[arg-type]
    points = matched_corners.astype(np.int32)
    cv2.fillConvexPoly(mask, points, (255,))
    save_debug_image(mask, title="mask")
    # Extract the region of interest using the mask
    x_min = int(max(0, np.floor(x_min)))
    x_max = int(min(image.shape[1], np.ceil(x_max)))
    y_min = int(max(0, np.floor(y_min)))
    y_max = int(min(image.shape[0], np.ceil(y_max)))
    roi = image_rotated[y_min:y_max, x_min:x_max]

    # Apply mask to the region of interest
    mask_roi = mask[y_min:y_max, x_min:x_max]
    matched_image = cv2.bitwise_and(roi, mask_roi)

    # Optionally, crop the matched region tightly around the template
    # You can also warp the matched region to align it with the template orientation
    save_debug_image(matched_image, title="matched region")
    return matched_image
