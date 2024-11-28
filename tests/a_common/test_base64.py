import numpy as np
import pytest

from codenames_parser.common.impr.reader import read_image
from codenames_parser.common.utils.base64 import base64_to_image, image_to_base64
from tests.fixtures import get_fixture_path
from tests.utils import list_diff, print_diff

TEST_CASES = [
    get_fixture_path("color_maps/classic/small_1.png"),
    get_fixture_path("boards/heb/board3_top.jpg"),
]


@pytest.mark.parametrize("fixture_file", TEST_CASES)
def test_b64_conversion(fixture_file: str):
    image_path = get_fixture_path(fixture_file)
    image = read_image(image_path)
    b64_string = image_to_base64(image)
    image_reconstructed = base64_to_image(b64_string)

    params_expected = _extract_image_params(image)
    params_actual = _extract_image_params(image_reconstructed)

    diff = list_diff(params_expected, params_actual, max_diff=0.5)
    print_diff(diff)
    assert not diff


def _extract_image_params(image: np.ndarray) -> list:
    mean_red = _rounded_color(image, 0)
    mean_green = _rounded_color(image, 1)
    mean_blue = _rounded_color(image, 2)
    return [
        str(image.shape),
        mean_red,
        mean_green,
        mean_blue,
        _rounded_float(image.std()),
    ]


def _rounded_color(image: np.ndarray, color_idx: int) -> float:
    return _rounded_float(image[:, :, color_idx].mean())


def _rounded_float(value, precision: int = 5) -> float:
    return round(float(value), precision)
