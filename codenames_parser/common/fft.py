import numpy as np

from codenames_parser.common.debug_util import save_debug_image
from codenames_parser.common.general import ensure_grayscale, grayscale_normalize


def calculate_fft(values: np.ndarray) -> np.ndarray:
    grayscale = ensure_grayscale(values)
    fft_result = np.fft.fft2(grayscale)
    fft_shifted = np.fft.fftshift(fft_result)
    fft_abs = np.log(np.abs(fft_shifted) + 1)
    fft_image = grayscale_normalize(fft_abs)
    save_debug_image(fft_image, title="fft")
    return fft_abs
