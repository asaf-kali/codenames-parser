import os
from shutil import copyfile
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sol4_utils
from scipy.ndimage import center_of_mass, label, map_coordinates
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
from scipy.signal import convolve2d
from tqdm import tqdm

from codenames_parser.common.debug_util import save_debug_image

DERIVATIVE_KERNEL = np.array([1, 0, -1]).reshape((1, 3))


def _partial_derivatives(im: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = convolve2d(im, DERIVATIVE_KERNEL, mode="same", boundary="symm")
    dy = convolve2d(im, DERIVATIVE_KERNEL.T, mode="same", boundary="symm")
    dxdy = dx * dy
    return dx, dy, dxdy


def _blur_multiple(*args: np.ndarray) -> Iterable[np.ndarray]:
    return [sol4_utils.blur_spatial(mat, 2) for mat in args]


def _calculate_response(dx: np.ndarray, dy: np.ndarray, dxy: np.ndarray, k: float = 0.01) -> np.ndarray:
    dx, dy, dxy = dx**2, dy**2, dxy**2
    det = dx * dy - dxy
    trace = dx + dy
    response = det - k * (trace**2)
    return response


def harris_corner_detector(im: np.ndarray) -> np.ndarray:
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    dx, dy, dxy = _partial_derivatives(im)
    dx, dy, dxy = _blur_multiple(dx, dy, dxy)
    response = _calculate_response(dx, dy, dxy)
    maximums = non_maximum_suppression(response)
    ret = np.argwhere(maximums.T)
    return ret


def _extract_descriptor(img: np.ndarray, corner: np.ndarray, offsets: np.ndarray, side: int) -> np.ndarray:
    """
    Extracts the descriptor around corner in the given image.
    :param img: the 3-rd level layer of a gaussian pyramid of an image.
    :param corner: the (x, y) coordinates of the corner to get the matrix from.
    :param offsets: an ndarray representing all offsets from the corner to sample as the output descriptor.
    :return: a normalized matrix of the region of img around corner of shape side x side.
    """
    coordinates = offsets + corner
    descriptor = map_coordinates(img.T, coordinates.T, order=1, prefilter=False)
    diff = descriptor - descriptor.mean()
    if min(descriptor) != max(descriptor):  # Safety check before division
        descriptor = diff / np.linalg.norm(diff)
    return descriptor.reshape((side, side))


def _build_offsets(desc_rad: int) -> Tuple[np.array, int]:
    rng = np.arange(start=-desc_rad, stop=desc_rad + 1)
    side = 1 + 2 * desc_rad
    result = []
    for i in range(side):
        for j in range(side):
            result.append([rng[i], rng[j]])
    return np.array(result), side


def sample_descriptor(im: np.ndarray, pos: np.ndarray, desc_rad: int) -> np.ndarray:
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the i'th descriptor at desc[i,:,:]. The per-descriptor dimensions KxK
             are related to the desc rad argument as follows K = 1+2*desc_rad.
    """
    offsets, side = _build_offsets(desc_rad)
    descriptors = []
    for i in range(pos.shape[0]):
        descriptor = _extract_descriptor(im, pos[i, :], offsets, side)
        descriptors.append(descriptor)
    return np.array(descriptors)


def find_features(pyr: List[np.ndarray]) -> List[np.ndarray]:
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    assert len(pyr) == 3
    orig, top = pyr[0], pyr[2]
    corners = spread_out_corners(orig, m=5, n=5, radius=15)
    descriptors = sample_descriptor(top, corners / 4, desc_rad=3)
    return [corners, descriptors]


def _flatten_descriptors(desc: np.ndarray) -> np.ndarray:
    return desc.reshape((desc.shape[0], desc.shape[1] * desc.shape[2]))


def _stack_masks(masks: List[np.ndarray]) -> np.ndarray:
    """
    Given a list of boolean matrix masks, returns a new mask where all given masks are true.
    """
    assert len(masks) > 0
    result = masks.pop(0)
    for mask in masks:
        result = np.logical_and(result, mask)
    return result


def match_features(desc1: np.ndarray, desc2: np.ndarray, min_score: float) -> List[np.ndarray]:
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    desc1 = _flatten_descriptors(desc1)
    desc2 = _flatten_descriptors(desc2)
    scores = desc1 @ desc2.T  # Rows is desc1 index, cols is desc2 index
    # Get the index of the 2nd best candidate for each column and row
    cols_2nd_best = np.argpartition(scores, -2, axis=0)[-2, :]
    rows_2nd_best = np.argpartition(scores, -2, axis=1)[:, -2]
    # Create mask for scores by each filter
    score_mask = scores >= min_score
    cols_mask = scores >= scores[cols_2nd_best, range(scores.shape[1])]
    rows_mask = scores >= scores[range(scores.shape[0]), rows_2nd_best][..., np.newaxis]
    # Get finalists
    final_mask = _stack_masks([score_mask, cols_mask, rows_mask])
    chosen_descriptors = np.argwhere(final_mask)
    return [chosen_descriptors[:, 0], chosen_descriptors[:, 1]]


def apply_homography(pos1: np.ndarray, H12: np.ndarray) -> np.ndarray:
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    homogeneous = np.stack([pos1[:, 0], pos1[:, 1], np.linspace(start=1, stop=1, num=pos1.shape[0])])
    transformed = H12 @ homogeneous
    x2, y2, z2 = np.split(transformed, indices_or_sections=3)
    x2, y2 = x2 / z2, y2 / z2
    result = np.row_stack([x2, y2]).T
    return result


def _get_inliers(points1: np.ndarray, points2: np.ndarray, inlier_tol: float, homography: np.ndarray) -> np.ndarray:
    transformed = apply_homography(points1, homography)
    distance = np.linalg.norm(points2 - transformed, axis=1) ** 2
    inliers = np.argwhere(distance <= inlier_tol)[:, 0]
    return inliers


def ransac_homography(
    points1: np.ndarray, points2: np.ndarray, num_iter: int, inlier_tol: float, translation_only: bool = False
) -> List[np.ndarray]:
    """
    Computes homography between two sets of points using RANSAC.
    :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    N = points1.shape[0]
    assert points2.shape[0] == N
    sample_size = 1 if translation_only else 2
    best_inliers = []
    choices = np.random.choice(N, size=(num_iter, sample_size))
    for index in choices:
        p1, p2 = points1[index], points2[index]
        hom = estimate_rigid_transform(p1, p2, translation_only)
        inliers = _get_inliers(points1, points2, inlier_tol, hom)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    homography = estimate_rigid_transform(points1[best_inliers], points2[best_inliers], translation_only)
    return [homography, best_inliers]


def _draw_lines(point1: np.ndarray, point2: np.ndarray, mask: np.ndarray, color: str, alpha: float = 1):
    group1, group2 = point1[mask], point2[mask]
    x = group1[:, 0], group2[:, 0]
    y = group1[:, 1], group2[:, 1]
    plt.plot(x, y, mfc="r", c=color, lw=0.5, alpha=alpha, ms=2, marker="o", markeredgewidth=0.0)


def display_matches(im1: np.ndarray, im2: np.ndarray, points1: np.ndarray, points2: np.ndarray, inliers: np.ndarray):
    """
    Display matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :param points1: An array shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param points2: An array shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    full = np.hstack((im1, im2))
    plt.imshow(full, cmap="gray")
    points2[:, 0] += im1.shape[1]  # Shift points2 according to hstack
    outliers = np.array(list(set(range(points1.shape[0])) - set(inliers)))
    _draw_lines(points1, points2, outliers, "blue", alpha=0.3)
    _draw_lines(points1, points2, inliers, "yellow")
    plt.show()


def _accumulate(matrices: Iterable[np.ndarray], inv: bool = False) -> List[np.ndarray]:
    result = []
    current = np.eye(3)
    for mat in matrices:
        mat = np.linalg.pinv(mat) if inv else mat
        current = current @ mat
        current /= current[2, 2]
        result.append(current)
    return result


def accumulate_homographies(H_succesive: List[np.ndarray], m: int) -> Iterable[np.ndarray]:
    """
    Convert a list of succesive homographies to a list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography matrices where H_successive[i] is a homography which transforms points from
    coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to accumulate the given homographies.
    :return: A list of M 3x3 homography matrices, where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    former = _accumulate(reversed(H_succesive[:m]))
    former.reverse()
    latter = _accumulate(H_succesive[m:], True)
    return former + [np.eye(3)] + latter


def compute_bounding_box(homography: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    tl = homography @ np.array([0, 0, 1])
    br = homography @ np.array([w, h, 1])
    return np.stack([tl[:2], br[:2]]).astype(np.int)


def _backwards_warping_coordinates(image: np.ndarray, homography: np.ndarray) -> Tuple[np.ndarray, int, int]:
    tl, br = compute_bounding_box(homography, image.shape[1], image.shape[0])
    x_axis, y_axis = np.arange(tl[0], br[0]), np.arange(tl[1], br[1])
    coordinates = np.stack(np.meshgrid(x_axis, y_axis)).T
    (
        height,
        width,
    ) = (
        coordinates.shape[0],
        coordinates.shape[1],
    )
    coordinates = coordinates.reshape((width * height, 2))
    coordinates = apply_homography(coordinates, np.linalg.inv(homography))
    return coordinates, height, width


def warp_channel(image: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    coordinates, height, width = _backwards_warping_coordinates(image, homography)
    warped = map_coordinates(image.T, coordinates.T, order=1, prefilter=False)
    warped = warped.reshape((height, width)).T
    return warped


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im: np.ndarray, m: int, n: int, radius: int) -> np.ndarray:
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j] : y_bound[j + 1], x_bound[i] : x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = (
        (corners[:, 0] > radius)
        & (corners[:, 0] < im.shape[1] - radius)
        & (corners[:, 1] > radius)
        & (corners[:, 1] < im.shape[0] - radius)
    )
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, max_images: int = None, fps=20):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param max_images: number of images to produce the panoramas with.
        """
        self.file_prefix = file_prefix
        self.files = [entry.path for entry in os.scandir(data_dir) if entry.path.endswith(".jpg")]
        self.files.sort()
        if max_images and max_images < len(self.files):
            self.files = self.files[:max_images]
        self.panoramas = None
        self.homographies = None
        self.fps = fps
        self.h = self.w = None
        print(f"Found {len(self.files)} images")

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in tqdm(self.files, desc="Building pyramids"):
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in tqdm(range(len(points_and_descriptors) - 1), desc="Create homographies"):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, 0.7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the
        # middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = (warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2
        x_strip_boundary = np.hstack(
            [np.zeros((number_of_panoramas, 1)), x_strip_boundary, np.ones((number_of_panoramas, 1)) * panorama_size[0]]
        )
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in tqdm(
            enumerate(self.frames_for_panoramas), "Create panoramas", total=self.frames_for_panoramas.shape[0]
        ):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i : i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset : boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0] : x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        if crop_left < crop_right:
            # assert crop_left < crop_right, "for testing your code with a few images do not crop."
            print(crop_left, crop_right)
            self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self, with_reverse=False):
        assert self.panoramas is not None
        out_folder = f"panoramic_frames/{self.file_prefix}"
        try:
            os.makedirs(out_folder)
        except:
            pass
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        total_frames = self.panoramas.shape[0]
        for i, panorama in tqdm(enumerate(self.panoramas), desc="Save panoramas", total=self.panoramas.shape[0]):
            panorama = (panorama * 255).astype(np.uint8)
            path = save_debug_image(panorama, title=f"Panorama {i + 1}")
            save_to = f"{out_folder}/panorama{i + 1:02d}.png"
            copyfile(path, save_to)
            if with_reverse:
                copyfile(save_to, f"{out_folder}/panorama{2 * total_frames - i:02d}.png")
        if os.path.exists(f"{self.file_prefix}.mp4"):
            os.remove(f"{self.file_prefix}.mp4")
        # write output video to current folder
        os.system(f"ffmpeg -framerate {self.fps} -i {out_folder}/panorama%02d.png {self.file_prefix}.mp4")

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()
