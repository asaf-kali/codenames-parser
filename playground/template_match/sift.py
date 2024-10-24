import cv2
import numpy as np

from codenames_parser.common.debug_util import draw_points, save_debug_image
from codenames_parser.common.general import ensure_grayscale, normalize, zero_pad
from codenames_parser.common.models import Point


def search_template_sift(source_image: np.ndarray, template_image: np.ndarray) -> np.ndarray:
    # Applying SIFT detector
    source_gray = ensure_grayscale(source_image)
    template_gray = ensure_grayscale(template_image)
    template_padded = zero_pad(template_gray, padding=30)

    source_features = _detect_features(source_gray)
    template_features = _detect_features(template_padded)

    source_kp = _detect_sift(source_gray)
    template_kp = _detect_sift(template_padded)

    return np.ndarray([])


def _detect_features(image: np.ndarray) -> np.ndarray:
    equalized = normalize(image)
    features = cv2.goodFeaturesToTrack(equalized, maxCorners=100, qualityLevel=0.01, minDistance=10)
    corners = cv2.cornerHarris(equalized, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    corners = cv2.dilate(corners, None)
    feature_points = [Point(int(feature[0][0]), int(feature[0][1])) for feature in features]
    draw_points(equalized, feature_points, title="features")
    # Threshold for an optimal value, it may vary depending on the image.
    corner_thresh = 0.01 * corners.max()
    is_corner = corners > corner_thresh
    equalized[is_corner] = 255
    save_debug_image(equalized, title="corners")
    return corners


def _detect_sift(image: np.ndarray) -> list[cv2.KeyPoint]:
    sift = cv2.SIFT_create()  # type: ignore
    key_points = list(sift.detect(image, None))
    marked = cv2.drawKeypoints(image, key_points, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    save_debug_image(marked, title="key points")
    return key_points
