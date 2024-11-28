import numpy as np

from codenames_parser.common.impr.color_manipulation import ensure_grayscale, normalize


def calculate_fft(values: np.ndarray) -> np.ndarray:
    grayscale = ensure_grayscale(values)
    fft_result = np.fft.fft2(grayscale)
    fft_shifted = np.fft.fftshift(fft_result)
    fft_abs = np.log(np.abs(fft_shifted) + 1)
    normalize(fft_abs, title="fft")
    return fft_abs
