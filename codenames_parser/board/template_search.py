import logging
from dataclasses import dataclass

import cv2
import numpy as np

from codenames_parser.common.align import apply_rotation
from codenames_parser.common.debug_util import save_debug_image
from codenames_parser.common.general import ensure_grayscale, has_larger_dimension
from codenames_parser.common.models import Point
from codenames_parser.common.scale import downsample_image

log = logging.getLogger(__name__)


@dataclass
class MatchResult:
    template: np.ndarray
    result_image: np.ndarray
    location: Point
    grade: float

    @classmethod
    def empty(cls):
        return cls(
            template=np.zeros((1, 1), dtype=np.uint8),
            result_image=np.zeros((1, 1), dtype=np.uint8),
            location=Point(0, 0),
            grade=-np.inf,
        )


@dataclass
class SearchResult:
    angle: float
    scale: float
    match: MatchResult

    @classmethod
    def empty(cls):
        return cls(angle=0.0, scale=0.0, match=MatchResult.empty())


def search_template(source_image: np.ndarray, template_image: np.ndarray, num_iterations: int = 2) -> np.ndarray:
    """Search for the template location in the source image using pyramid search.

    Args:
        source_image (np.ndarray): Source image.
        template_image (np.ndarray): Template image.
        num_iterations (int, optional): Number of iterations.

    Returns:
        np.ndarray: Matched region from the source image.
    """
    # Convert to grayscale
    source_gray = ensure_grayscale(source_image)
    # source_gray = border_pad(source_gray, padding=source_gray.shape[0] // 10)
    template_gray = ensure_grayscale(template_image)
    # Angle and scale ranges
    scale_ratio = max(template_image.shape[0] / source_image.shape[0], template_image.shape[1] / source_image.shape[1])
    min_angle, max_angle = (-5, 5)
    min_scale, max_scale = 0.1, round(1.0 / scale_ratio, 4)
    angle_step_num = 6
    scale_step_num = 4
    iter_angles = np.linspace(min_angle, max_angle, num=angle_step_num * 2 + 1)
    iter_scales = np.linspace(min_scale, max_scale, num=scale_step_num * 2 + 1)
    # Initial best values
    search_result = SearchResult.empty()

    # Iterate
    for i in range(1, num_iterations + 1):
        # Downsample factor
        factor = 2 ** (num_iterations - i)
        log.info(f"Iteration {i}: downsample factor={factor}")
        source_gray = apply_rotation(image=source_gray, angle_degrees=-search_result.angle)
        source_downsample = downsample_image(source_gray, factor=factor)
        save_debug_image(source_downsample, title=f"source downsample {i}")

        iteration_search_result = SearchResult.empty()
        # For each angle and scale
        for angle in iter_angles:
            for scale in iter_scales:
                if scale > max_scale:
                    continue
                # Transform template
                template_transformed = _transform_template(template_gray, angle, scale, factor=factor)
                # save_debug_image(template_transformed, title=f"template transformed {i} ({angle:.2f}°, X{scale:.2f})")
                if has_larger_dimension(template_transformed, source_downsample):
                    continue
                # Perform template matching
                match_result = _match_template(source=source_downsample, template=template_transformed)
                # save_debug_image(match_result.result_image, title=f"match result {match_result.grade:.3f}")
                # Update best match if necessary
                if match_result.grade > iteration_search_result.match.grade:
                    iteration_search_result = SearchResult(angle=angle, scale=scale, match=match_result)

        _log_iteration(i, result=iteration_search_result)
        search_result = iteration_search_result

    matched_image = _crop_best_result(
        source_gray,
        best_angle=search_result.angle,
        best_location=search_result.match.location,
        best_template=search_result.match.template,
    )
    return matched_image


def _transform_template(template: np.ndarray, angle: float, scale: float, factor: int) -> np.ndarray:
    """Rotate and scale the template image, including downsampling factor, with minimal padding to prevent pixel loss.

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

    # Get the center of the image
    center = (new_width / 2, new_height / 2)

    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute the sine and cosine of the rotation angle
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    # Compute the new bounding dimensions of the image
    bound_w = int(new_height * abs_sin + new_width * abs_cos)
    bound_h = int(new_height * abs_cos + new_width * abs_sin)

    # Adjust the rotation matrix to account for translation
    rotation_matrix[0, 2] += bound_w / 2 - center[0]
    rotation_matrix[1, 2] += bound_h / 2 - center[1]

    # Perform the rotation with the adjusted matrix
    rotated_template = cv2.warpAffine(
        resized_template,
        rotation_matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return rotated_template


def _match_template(source: np.ndarray, template: np.ndarray) -> MatchResult:
    """Perform template matching using normalized cross-correlation.

    Args:
        source (np.ndarray): Source image.
        template (np.ndarray): Template image.

    Returns:
        np.ndarray: Matching result.
    """
    match_result = cv2.matchTemplate(source, template, method=cv2.TM_CCOEFF_NORMED)
    _, _, _, peak_coords = cv2.minMaxLoc(match_result)
    peak_point = Point(peak_coords[0], peak_coords[1])
    grade = _compute_psr(match_result, peak_point=peak_point)
    result_image = cv2.normalize(match_result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # type: ignore[call-overload]
    point = Point(peak_coords[0], peak_coords[1])
    return MatchResult(template=template, location=point, grade=grade, result_image=result_image)


def _compute_psr(match_result: np.ndarray, peak_point: Point) -> float:
    """Compute the Peak-to-Sidelobe Ratio (PSR).

    Args:
        match_result (np.ndarray): Result from template matching.
        peak_point (Point): Coordinates of the peak correlation value.

    Returns:
        float: PSR value.
    """
    if peak_point == (0, 0):
        return -np.inf
    peak_value = match_result[peak_point[1], peak_point[0]]

    # Exclude a small region around the peak
    mask = np.ones_like(match_result, dtype=bool)
    h, w = match_result.shape
    peak_x, peak_y = peak_point
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


def _log_iteration(i: int, result: SearchResult):
    match = result.match
    save_debug_image(match.template, title=f"best template {i} ({result.angle:.2f}°, X{result.scale:.2f})")
    save_debug_image(match.result_image, title=f"best match {i} ({match.grade:.3f})")
    log.info(f"Iteration {i}: angle={result.angle:<6.2f} scale={result.scale:<6.2f} value={match.grade:<6.3f}")
