import logging
from dataclasses import dataclass

import cv2
import numpy as np

from codenames_parser.common.align import apply_rotation
from codenames_parser.common.crop import rotated_crop
from codenames_parser.common.debug_util import save_debug_image
from codenames_parser.common.general import (
    ensure_grayscale,
    has_larger_dimension,
    normalize,
)
from codenames_parser.common.models import Point
from codenames_parser.common.scale import downsample_image

log = logging.getLogger(__name__)


@dataclass
class MatchResult:
    template: np.ndarray
    convo: np.ndarray
    convo_normalized: np.ndarray
    location: Point
    score: float

    @property
    def min_value(self):
        return np.min(self.convo)

    @property
    def max_value(self):
        return np.max(self.convo)


@dataclass
class SearchResult:
    angle: float
    scale: float
    match: MatchResult

    @property
    def name(self):
        return f"{self.angle:.2f}°, X{self.scale:.2f}"

    def __str__(self) -> str:
        return f"{self.name} score={self.match.score:.3f}"


def search_template(source_image: np.ndarray, template_image: np.ndarray, num_iterations: int = 1) -> np.ndarray:
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
    angle_step_num = 5
    scale_step_num = 3
    iter_angles = np.linspace(min_angle, max_angle, num=angle_step_num * 2 + 1)
    iter_scales = np.linspace(min_scale, max_scale, num=scale_step_num * 2 + 1)

    search_result = None
    # Iterate
    for i in range(1, num_iterations + 1):
        # Downsample factor
        factor = 2 ** (num_iterations - i)
        log.info(f"Iteration {i}: downsample factor={factor}")
        angle_degrees = _get_rotation_angle(search_result)
        source_gray = apply_rotation(image=source_gray, angle_degrees=angle_degrees)
        source_downsample = downsample_image(source_gray, factor=factor)
        save_debug_image(source_downsample, title=f"source downsample {i}")

        # For each angle and scale
        iteration_results = []
        for angle in iter_angles:
            for scale in iter_scales:
                if scale > max_scale:
                    log.debug(f"Skipping scale {scale:.2f}")
                    continue
                # Transform template
                template_transformed = _transform_template(template_gray, angle, scale, factor=factor)
                if has_larger_dimension(template_transformed, source_downsample):
                    log.debug(f"Skipping template {template_transformed.shape} larger than source")
                    continue
                # save_debug_image(template_transformed, title=f"template transformed {i} ({angle:.2f}°, X{scale:.2f})")
                # Perform template matching
                match_result = _match_template(source=source_downsample, template=template_transformed)
                log.debug(f"angle={angle:<6.2f} scale={scale:<6.2f} score={match_result.score:<6.3f}")
                # save_debug_image(match_result.result_normalized, title=f"match result {match_result.grade:.3f}")
                iteration_result = SearchResult(angle=round(angle, 3), scale=round(scale, 3), match=match_result)
                iteration_results.append(iteration_result)

        # Find the best result
        best_iteration_result = _pick_best_result(iteration_results)
        # _log_iteration(i, result=best_iteration_result)
        search_result = best_iteration_result
        # _plot_results(iteration_results)
    if search_result is None:
        raise ValueError("No match found")
    matched_image = rotated_crop(
        source_gray,
        angle=search_result.angle,
        top_left=search_result.match.location,
        size=search_result.match.template.shape[:2],
    )
    return matched_image


def _pick_best_result(iteration_results: list[SearchResult]) -> SearchResult:
    results_ordered = sorted(iteration_results, key=lambda x: x.match.score, reverse=True)
    log.info("Top 5 results:")
    for j in range(5):
        result = results_ordered[j]
        log.info(str(result))
        save_debug_image(result.match.template, title=f"template {j} ({result.name})")
        save_debug_image(result.match.convo_normalized, title=f"match {j} ({result.name})")
    best_iteration_result = results_ordered[0]
    return best_iteration_result


def _match_template(source: np.ndarray, template: np.ndarray) -> MatchResult:
    log.debug(f"Matching template {template.shape} in source {source.shape}")
    match_result = cv2.matchTemplate(source, template, method=cv2.TM_CCOEFF_NORMED)
    result_normalized = normalize(match_result)
    _, _, _, peak_coords = cv2.minMaxLoc(match_result)
    peak_point = Point(peak_coords[0], peak_coords[1])
    if np.multiply(*match_result.shape) < 50:
        score = 0.0
    else:
        score = _grade_match(match_result, result_normalized, peak_point, template.shape[:2])
    return MatchResult(
        template=template,
        convo=match_result,
        convo_normalized=result_normalized,
        location=peak_point,
        score=score,
    )


def _grade_match(
    match_result: np.ndarray, result_normalized: np.ndarray, peak_point: Point, template_size: tuple[int, int]
) -> float:
    p = float(np.percentile(result_normalized, q=75))
    p_normal = 1 - p / 255  # Higher is better
    psr = _calculate_psr(match_result, peak_point)  # Higher is better
    area_factor = _calculate_area_factor(template_size)
    score = 300 * p_normal + 2 * psr * area_factor
    log.debug(f"p_normal={p_normal:<5.3f} psr={psr:<5.3f} area_factor={area_factor:<5.3f} score={score:<5.3f}")
    return score


def _calculate_area_factor(template_size: tuple[int, int]) -> float:
    template_area = np.multiply(*template_size)
    area_log = np.log(template_area)
    # area_factor = 1 / (1 + area_log)
    return area_log


def _calculate_psr(match_result: np.ndarray, peak_point: Point) -> float:
    peak_value = match_result[peak_point[1], peak_point[0]]
    # Calculate the mean value of the sidelobe
    sidelobe = np.copy(match_result)
    sidelobe[peak_point[1], peak_point[0]] = 0
    mean_sidelobe = np.mean(sidelobe)
    # Calculate the standard deviation of the sidelobe
    std_sidelobe = np.std(sidelobe)
    # Calculate the peak-to-sidelobe
    psr = (peak_value - mean_sidelobe) / std_sidelobe
    return float(psr)


def _transform_template(template: np.ndarray, angle: float, scale: float, factor: int) -> np.ndarray:
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


def _get_rotation_angle(search_result: SearchResult | None) -> float:
    if search_result is None:
        return 0
    return -search_result.angle


def _log_iteration(i: int, result: SearchResult):
    match = result.match
    save_debug_image(match.template, title=f"best template {i} ({result.angle:.2f}°, X{result.scale:.2f})")
    save_debug_image(match.convo_normalized, title=f"best match {i} ({match.max_value:.3f})")
    log.info(f"Iteration {i}: angle={result.angle:<6.2f} scale={result.scale:<6.2f} score={match.score:<6.3f}")


# def _plot_results(results: list[SearchResult]):
#     plt.figure()
#     # plt.xlim(0, 100)
#     # plt.ylim(0, 20)
#     plt.xlabel("P factor")
#     plt.ylabel("PSR")
#     # Give each point a label
#     # for result in results:
#     # grade = result.match.score
#     # x, y = grade
#     # plt.text(x, y, result.name)
#     # plt.scatter(x, y)
#     plt.legend()
#     plt.show()
