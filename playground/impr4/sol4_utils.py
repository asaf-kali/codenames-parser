from typing import List, Tuple

import numpy as np
import sol3
from scipy.signal import convolve2d


def build_gaussian_pyramid(im: np.ndarray, max_levels: int, filter_size: int) -> Tuple[List[np.ndarray], np.ndarray]:
    return sol3.build_gaussian_pyramid(im, max_levels, filter_size)


def pyramid_blending(
    im1: np.ndarray, im2: np.ndarray, mask: np.ndarray, max_levels: int, filter_size_im: int, filter_size_mask: int
) -> np.ndarray:
    return sol3.pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask)


def gaussian_kernel(kernel_size: int) -> np.ndarray:
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, "full")
    return kernel / kernel.sum()


def blur_spatial(img: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, "same", "symm")
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, "same", "symm")
    return blur_img
