import logging

import pytest
from codenames.classic.color import ClassicColor

from codenames_parser.color_map.color_map_parser import parse_color_map
from codenames_parser.common.image_reader import read_image
from tests.a_color_map.cases import MAP_CASES
from tests.fixtures import get_fixture_path
from tests.utils import list_diff, print_diff

log = logging.getLogger(__name__)


@pytest.mark.parametrize("fixture_file,expected_colors", MAP_CASES)
def test_parse_color_map(fixture_file: str, expected_colors: list[ClassicColor]):
    image_path = get_fixture_path(f"color_maps/classic/{fixture_file}")
    image = read_image(image_path)
    colors = parse_color_map(image=image, color_type=ClassicColor)
    diff = list_diff(l1=expected_colors, l2=colors)
    print_diff(diff)
    assert not diff
