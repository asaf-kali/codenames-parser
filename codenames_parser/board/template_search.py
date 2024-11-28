import logging
from dataclasses import dataclass

import cv2
import numpy as np

from codenames_parser.common.impr.align import apply_rotation
from codenames_parser.common.impr.color_manipulation import normalize
from codenames_parser.common.impr.crop import rotated_crop
from codenames_parser.common.impr.padding import border_pad
from codenames_parser.common.impr.scale import downsample_image, has_larger_dimension
from codenames_parser.common.impr.transform import transform
from codenames_parser.common.utils.debug_util import save_debug_image
from codenames_parser.common.utils.models import Point

log = logging.getLogger(__name__)

ANGLE_STEP_NUM = 5
SCALE_STEM_NUM = 3
ITERATION_ANGLE_DIFF = 2.0
ITERATION_SCALE_DIFF = 0.3


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


def search_template(source: np.ndarray, template: np.ndarray, num_iterations: int = 3) -> np.ndarray:
    """Search for the template location in the source image using pyramid search.

    Args:
        source (np.ndarray): Source image.
        template (np.ndarray): Template image.
        num_iterations (int, optional): Number of iterations.

    Returns:
        np.ndarray: Matched region from the source image.
    """
    source = border_pad(source, padding=5)
    # Angle and scale ranges
    scale_ratio = max(template.shape[0] / source.shape[0], template.shape[1] / source.shape[1])
    max_possible_scale = round(1.0 / scale_ratio, 4)
    min_angle, max_angle = (-10.0, 10.0)
    min_scale, max_scale = 0.1, max_possible_scale

    search_result = None
    # Iterate
    for i in range(1, num_iterations + 1):
        # Iteration parameters
        iter_angles = _rounded_list(np.linspace(min_angle, max_angle, num=ANGLE_STEP_NUM * 2 + 1))
        iter_scales = _rounded_list(np.linspace(min_scale, max_scale, num=SCALE_STEM_NUM * 2 + 1))
        factor = 2 ** (num_iterations - i)
        _log_iteration_search_params(i, factor, iter_angles, iter_scales)
        # Adjust the source image
        angle_degrees = _get_rotation_angle(search_result)
        source = apply_rotation(image=source, angle_degrees=angle_degrees)
        # Run iteration search
        iteration_results = _template_search_iteration(
            i=i, source=source, template=template, factor=factor, angles=iter_angles, scales=iter_scales
        )
        # Find the best result
        best_iteration_result = _pick_best_result(iteration_results)
        _log_iteration_result(i, result=best_iteration_result)
        search_result = best_iteration_result
        # Narrow the search range
        min_angle, max_angle = min_angle / 2, max_angle / 2
        min_scale, max_scale = search_result.scale - ITERATION_SCALE_DIFF, search_result.scale + ITERATION_SCALE_DIFF
        max_scale = min(max_scale, max_possible_scale)
    # Final search result
    if search_result is None:
        raise ValueError("No match found")
    matched_image = rotated_crop(
        source,
        angle=search_result.angle,
        top_left=search_result.match.location,
        size=search_result.match.template.shape[:2],
    )
    log.debug(f"Final matching result: {search_result}")
    return matched_image


def _template_search_iteration(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    i: int, source: np.ndarray, template: np.ndarray, factor: int, angles: list[float], scales: list[float]
) -> list[SearchResult]:
    source_downsample = downsample_image(source, factor=factor)
    save_debug_image(source_downsample, title=f"source downsample {i}")
    # For each angle and scale
    iteration_results = []
    for angle in angles:
        for scale in scales:
            # Transform template
            template_transformed = transform(image=template, scale=scale / factor, angle=angle)
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
    return iteration_results


def _grade_match(
    match_result: np.ndarray, result_normalized: np.ndarray, peak_point: Point, template_size: tuple[int, int]
) -> float:
    p = float(np.percentile(result_normalized, q=75))
    p_normal = 1 - p / 255  # Higher is better
    max_value = match_result[peak_point[1], peak_point[0]]
    psr = _calculate_psr(match_result, peak_point)  # Higher is better
    area_factor = _calculate_area_factor(template_size)
    score = (300 * p_normal + 100 * max_value + 5 * psr) * area_factor
    # max_value_factored = max_value * area_factor
    # score = max_value_factored / area_factor
    log.debug(f"p_normal={p_normal:<5.3f} psr={psr:<5.3f} area_factor={area_factor:<5.3f} score={score:<5.3f}")
    return float(score)


def _pick_best_result(iteration_results: list[SearchResult]) -> SearchResult:
    results_ordered = sorted(iteration_results, key=lambda x: x.match.score, reverse=True)
    log.debug("Top results:")
    for j in range(3):
        result = results_ordered[j]
        log.debug(str(result))
        # save_debug_image(result.match.template, title=f"template {j} ({result.name})")
        # save_debug_image(result.match.convo_normalized, title=f"match {j} ({result.name})")
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


def _calculate_area_factor(template_size: tuple[int, int]) -> float:
    template_area = np.multiply(*template_size)
    area_log = np.log(template_area)
    return area_log


def _get_rotation_angle(search_result: SearchResult | None) -> float:
    if search_result is None:
        return 0
    return -search_result.angle


def _log_iteration_search_params(i: int, factor: int, iter_angles: list[float], iter_scales: list[float]) -> None:
    log.debug(f"Iteration {i}")
    log.debug(f"Angles: {iter_angles}")
    log.debug(f"Scales: {iter_scales}")
    log.debug(f"Downsample factor: {factor}")


def _log_iteration_result(i: int, result: SearchResult):
    match = result.match
    save_debug_image(match.template, title=f"best template {i} ({result.angle:.2f}°, X{result.scale:.2f})")
    # save_debug_image(match.convo_normalized, title=f"best match {i} ({match.score:.3f})")
    log.info(f"Iteration {i}: angle={result.angle:<6.2f} scale={result.scale:<6.2f} score={match.score:<6.3f}")


def _rounded_list(vector: np.ndarray) -> list[float]:
    return [round(float(x), 3) for x in vector]
