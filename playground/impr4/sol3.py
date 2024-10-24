import os
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2 import imread
from scipy.ndimage import convolve

MAX_COLOR = 255
COLOR_NUM = MAX_COLOR + 1
GRAY_CODE = 1
RGB_CODE = 2
MIN_RES = 32
COLOR_DIM = 3
FILTER_BASE = np.array([1, 1])


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def read_image(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image in a given representation.
    :param filename: the path to the file to read.
    :param representation: the file representation (1 for grayscale, 2 for rgb).
    :return: an ndarray of float64 representing the file. The values are normalized at range [0, 1].
    """
    assert representation == GRAY_CODE or representation == RGB_CODE
    as_gray = representation == GRAY_CODE
    img = imread(filename)
    if img.max() > 1:
        img = img / MAX_COLOR
    if as_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _show(img: np.ndarray):
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.show()


def _build_filter_vec(size: int) -> np.ndarray:
    if size == 1:
        return np.array([[1]])
    assert size % 2 != 0
    filter_vec = FILTER_BASE
    for i in range(size - 2):
        filter_vec = np.convolve(filter_vec, FILTER_BASE)
    filter_vec = filter_vec / filter_vec.sum()
    return filter_vec.reshape((1, size))


def _blur(img: np.ndarray, filter_vec: np.ndarray) -> np.ndarray:
    img = convolve(img, filter_vec)
    img = convolve(img, filter_vec.T)
    return img


def _down_sample(img: np.ndarray) -> np.ndarray:
    return img[::2, ::2]


def _is_too_small(img: np.ndarray) -> bool:
    return min(img.shape) < MIN_RES


def build_gaussian_pyramid(im: np.ndarray, max_levels: int, filter_size: int) -> Tuple[List[np.ndarray], np.ndarray]:
    pyr = [im]
    filter_vec = _build_filter_vec(filter_size)
    for level in range(max_levels - 1):
        last = pyr[-1]
        if _is_too_small(last):
            break
        reduced = _blur(last, filter_vec)
        reduced = _down_sample(reduced)
        pyr.append(reduced)
    return pyr, filter_vec


def _expand(img: np.ndarray, filter_vec: np.ndarray) -> np.ndarray:
    expanded = np.zeros((img.shape[0] * 2, img.shape[1] * 2))
    expanded[::2, ::2] = img
    expanded = _blur(expanded, filter_vec * 2)
    return expanded


def build_laplacian_pyramid(im: np.ndarray, max_levels: int, filter_size: int) -> Tuple[List[np.ndarray], np.ndarray]:
    gaus, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = []
    for level in range(len(gaus) - 1):
        expanded = _expand(gaus[level + 1], filter_vec)
        lap = gaus[level] - expanded
        pyr.append(lap)
    pyr.append(gaus[-1])
    return pyr, filter_vec


def _strech(img: np.ndarray) -> np.ndarray:
    return (img - img.min()) / img.max()


def laplacian_to_image(lpyr: List[np.ndarray], filter_vec: np.ndarray, coeff: List[float]) -> np.ndarray:
    assert len(lpyr) == len(coeff)
    assert len(lpyr) > 0
    img = None
    for lap, c in zip(reversed(lpyr), reversed(coeff)):
        if img is None:
            img = np.zeros(lap.shape)
        else:
            img = _expand(img, filter_vec)
        img = img + lap * c
    return _strech(img)


def _get_base_image(pyr: List[np.ndarray]) -> np.ndarray:
    height, _ = pyr[0].shape
    width = sum(img.shape[1] for img in pyr)
    res = np.zeros((height, width))
    return res


def render_pyramid(pyr: List[np.ndarray], levels: int) -> np.ndarray:
    assert len(pyr) > 0
    assert levels <= len(pyr)
    pyr = pyr[:levels]
    res = _get_base_image(pyr)
    start, end = 0, 0
    for img in pyr:
        img = _strech(img)
        end = start + img.shape[1]
        res[0 : img.shape[0], start:end] = img
        start = end
    return res


def display_pyramid(pyr: List[np.ndarray], levels: int):
    res = render_pyramid(pyr, levels)
    plt.imshow(res, cmap="gray")
    plt.show()


def _combine_level(im1: np.ndarray, im2: np.ndarray, gm: np.ndarray) -> np.ndarray:
    return im1 * gm + im2 * (1 - gm)


def _generate_combined_pyramid(
    gm: List[np.ndarray], l1: List[np.ndarray], l2: List[np.ndarray], max_levels: int
) -> List[np.ndarray]:
    l_out = []
    assert len(l1) == len(l2) == len(gm)
    max_levels = min(max_levels, len(l1))
    for i in range(max_levels):
        combined = _combine_level(l1[i], l2[i], gm[i])
        l_out.append(combined)
    return l_out


def pyramid_blending(
    im1: np.ndarray, im2: np.ndarray, mask: np.ndarray, max_levels: int, filter_size_im: int, filter_size_mask: int
) -> np.ndarray:
    assert im1.shape == im2.shape == mask.shape, f"Should have the same shape: {im1.shape} = {im2.shape} = {mask.shape}"
    assert filter_size_im % 2 != 0
    assert filter_size_mask % 2 != 0
    l1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l2, filter_vec = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    gm, filter_vec_mask = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    l_out = _generate_combined_pyramid(gm, l2, l1, max_levels)
    combined = laplacian_to_image(l_out, filter_vec, [1 for _ in l_out])
    combined = combined.clip(min=0, max=1)
    return combined


def _img_source(name: str, code=RGB_CODE) -> np.ndarray:
    path = relpath(f"externals/{name}.jpg")
    return read_image(path, code)


def _mask_to_bool(mask: np.ndarray) -> np.ndarray:
    as_bool = np.ndarray(shape=(mask.shape[0], mask.shape[1]), dtype=np.bool)
    as_bool[mask != 0] = True
    as_bool[mask == 0] = False
    return as_bool


def _blend_rgb(im1: np.ndarray, im2: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.ndarray((im1.shape[0], im1.shape[1], 3))
    for i in range(COLOR_DIM):
        im1_channel = im1[:, :, i]
        im2_channel = im2[:, :, i]
        result_channel = pyramid_blending(im1_channel, im2_channel, mask, 5, 7, 7)
        result[:, :, i] = result_channel
    return result


def _display(images: Dict[str, np.ndarray]):
    margin = 50
    size = 200
    dpi = 100
    rows = cols = 2

    width = (len(images) * (size + margin)) / dpi  # inches
    height = (len(images) * (size + margin)) / dpi

    left = margin / dpi / width  # axes ratio
    bottom = margin / dpi / height

    fig, axes = plt.subplots(rows, cols, figsize=(width, height), dpi=dpi)
    fig.subplots_adjust(left=left, bottom=bottom, right=1 - left, top=1 - bottom)

    for ax, (title, image) in zip(axes.flatten(), images.items()):
        ax.axis("off")
        ax.set_title(title)
        if image.dtype == np.bool:
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image)
    plt.show()


def blending_example(im1: str, im2: str, mask: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    im1 = _img_source(im1)
    im2 = _img_source(im2)
    mask = _img_source(mask, GRAY_CODE)
    mask = _mask_to_bool(mask)
    result = _blend_rgb(im1, im2, mask)
    _display({"Image 1": im1, "Image 2": im2, "Mask": mask, "Result": result})
    return im1, im2, mask, result


def blending_example1() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return blending_example("view", "head", "mask1")


def blending_example2() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return blending_example("paris", "whale", "mask2")
