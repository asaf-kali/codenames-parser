import cv2
import numpy as np

from codenames_parser.common.impr.color_manipulation import ensure_grayscale
from codenames_parser.common.impr.general import apply_per_channel
from codenames_parser.common.utils.debug_util import save_debug_image
from codenames_parser.common.utils.models import Size

DEFAULT_GRID_SIZE = Size(width=8, height=8)


def contrast_limit_equalization(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: Size = DEFAULT_GRID_SIZE,
    title: str = "contrast limit equalized",
    save: bool = True,
) -> np.ndarray:
    equalized = apply_per_channel(
        image=image,
        func=_contrast_limit_equalization,
        clip_limit=clip_limit,
        grid_size=grid_size,
    )
    if save:
        save_debug_image(equalized, title=title)
    return equalized


def _contrast_limit_equalization(image: np.ndarray, clip_limit: float, grid_size: Size) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    equalized = clahe.apply(image)
    return equalized


# def local_histogram_equalization(
#     image: np.ndarray,
#     radius: int = 80,
#     title: str = "local equalized",
#     save: bool = True,
# ) -> np.ndarray:
#     equalized = apply_per_channel(
#         image=image,
#         func=_local_histogram_equalization,
#         radius=radius,
#     )
#     if save:
#         save_debug_image(equalized, title=title)
#     return equalized


# def _local_histogram_equalization(image: np.ndarray, radius: int) -> np.ndarray:
#     from skimage.filters import rank
#     from skimage.morphology import disk
#
#     footprint = disk(radius)
#     equalized = rank.equalize(image, footprint=footprint)
#     return equalized


def equalize_histogram(image: np.ndarray, save: bool = True) -> np.ndarray:
    image = ensure_grayscale(image)
    equalized = cv2.equalizeHist(image)
    if save:
        save_debug_image(equalized, title="equalized")
    return equalized
