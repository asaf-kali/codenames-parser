import logging

import pytest
from codenames.classic.color import ClassicColor
from codenames.duet.card import DuetColor

from codenames_parser.color_map.color_map_parser import parse_color_map
from codenames_parser.common.image_reader import read_image
from tests.a_color_map.cases import MAP_CASES
from tests.fixtures import get_fixture_path
from tests.utils import list_diff, print_diff

log = logging.getLogger(__name__)


def _get_fixture_folder_by_color_type(color_type: type[ClassicColor]) -> str:
    if color_type is ClassicColor:
        return "classic"
    if color_type is DuetColor:
        return "duet"
    raise NotImplementedError(f"Unknown color type: {color_type}")


def _get_color_map_path(fixture_file: str, color_type: type[ClassicColor]) -> str:
    folder = _get_fixture_folder_by_color_type(color_type=color_type)
    return get_fixture_path(f"color_maps/{folder}/{fixture_file}")


@pytest.mark.parametrize("fixture_file,expected_colors,color_type", MAP_CASES)
def test_parse_color_map(fixture_file: str, expected_colors: list[ClassicColor], color_type: type[ClassicColor]):
    image_path = _get_color_map_path(fixture_file=fixture_file, color_type=color_type)
    image = read_image(image_path)
    colors = parse_color_map(image=image, color_type=color_type)
    diff = list_diff(l1=expected_colors, l2=colors)
    print_diff(diff)
    assert not diff
