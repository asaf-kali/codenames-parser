import logging

import pytest
from codenames.game.color import CardColor

from codenames_parser.color_map.color_map_parser import parse_color_map
from tests.a_color_map.cases import MAP_CASES
from tests.fixtures import get_fixture_path
from tests.utils import list_diff, print_diff

log = logging.getLogger(__name__)


@pytest.mark.parametrize("fixture_file,expected_colors", MAP_CASES)
def test_parse_color_map(fixture_file: str, expected_colors: list[CardColor]):
    image_path = get_fixture_path(fixture_file)
    colors = parse_color_map(image_path=image_path)
    diff = list_diff(l1=expected_colors, l2=colors)
    print_diff(diff)
    assert not diff
