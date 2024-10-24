import logging

import cv2
import numpy as np

from codenames_parser.common.debug_util import save_debug_image
from codenames_parser.common.general import ensure_grayscale

log = logging.getLogger(__name__)


def detect_template_in_source(template: np.ndarray, source: np.ndarray):
    """
    Detect the scale, rotation, and position of the template in the source image.

    Parameters:
    template (np.ndarray): Template image.
    source (np.ndarray): Source image.

    Returns:
    tuple[float, float, tuple[float, float]]: Scale, rotation angle, and translation (tx, ty).
    """
    # Compute keypoints and descriptors
    template = ensure_grayscale(template)
    source = ensure_grayscale(source)
    kp1, des1 = compute_sift_keypoints_descriptors(template)
    kp2, des2 = compute_sift_keypoints_descriptors(source)
    # Match descriptors
    matches = match_descriptors(des1, des2)

    def draw_matches():
        img_matches = cv2.drawMatches(template, kp1, source, kp2, matches, None)
        save_debug_image(img_matches, title="matches")

    # Compute homography
    hmg = compute_homography(kp1, kp2, matches)

    # Extract scale, rotation, and translation
    scale, angle, translation = extract_scale_rotation_translation(hmg)
    log.info(f"Scale: {scale}, Angle: {angle}, Translation: {translation}")
    crop = cv2.warpPerspective(template, hmg, (source.shape[1], source.shape[0]))
    save_debug_image(crop, title="template in source")
    return scale, angle, translation, hmg, crop


def compute_sift_keypoints_descriptors(image: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """
    Compute SIFT keypoints and descriptors for an image.

    Parameters:
    image (np.ndarray): Grayscale image.

    Returns:
    tuple[list[cv2.KeyPoint], np.ndarray]: Keypoints and descriptors.
    """
    sift = cv2.SIFT_create()  # type: ignore
    keypoints, descriptors = sift.detectAndCompute(image, None)
    marked = cv2.drawKeypoints(image, keypoints, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    save_debug_image(marked, title="key points")
    return keypoints, descriptors


def match_descriptors(des1: np.ndarray, des2: np.ndarray, ratio_thresh: float = 0.75) -> list[cv2.DMatch]:
    """
    Match descriptors between two images using BFMatcher and Lowe's ratio test.

    Parameters:
    des1 (np.ndarray): Descriptors from the first image.
    des2 (np.ndarray): Descriptors from the second image.
    ratio_thresh (float): Threshold for Lowe's ratio test.

    Returns:
    list[cv2.DMatch]: list of good matches.
    """
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    return good_matches


def compute_homography(kp1: list[cv2.KeyPoint], kp2: list[cv2.KeyPoint], matches: list[cv2.DMatch]) -> np.ndarray:
    """
    Compute homography matrix using RANSAC.

    Parameters:
    kp1 (list[cv2.KeyPoint]): Keypoints from the first image.
    kp2 (list[cv2.KeyPoint]): Keypoints from the second image.
    matches (list[cv2.DMatch]): Matched keypoints.

    Returns:
    np.ndarray: Homography matrix.
    """
    if len(matches) < 4:
        raise ValueError("Not enough matches to compute homography.")
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    return homography


def extract_scale_rotation_translation(H: np.ndarray) -> tuple[float, float, tuple[float, float]]:
    """
    Extract scale, rotation angle (in degrees), and translation from homography matrix.

    Parameters:
    H (np.ndarray): Homography matrix.

    Returns:
    tuple[float, float, tuple[float, float]]: Scale, rotation angle, and translation (tx, ty).
    """
    # Normalize the matrix to ensure H[2,2] is 1
    H = H / H[2, 2]

    # Extract scale and rotation
    r1 = H[0:2, 0]
    r2 = H[0:2, 1]
    t = H[0:2, 2]

    scale_x = np.linalg.norm(r1)
    scale_y = np.linalg.norm(r2)
    scale = (scale_x + scale_y) / 2

    theta = np.arctan2(r1[1], r1[0])
    angle = np.degrees(theta)

    tx, ty = t[0], t[1]

    return scale, angle, (tx, ty)
