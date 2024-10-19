import logging
import os

import pytest
from codenames.game.color import CardColor

from codenames_parser.color_map.color_map_parser import parse_color_map
from codenames_parser.common.grid_detection import GRID_WIDTH
from codenames_parser.common.models import Grid
from tests.cases import MAP_CASES
from tests.fixtures import get_fixture_path
from tests.utils import print_diff

log = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def set_env_vars():
    os.environ["DEBUG_DISABLED"] = "true"


@pytest.mark.parametrize("fixture_file,expected_grid", MAP_CASES)
def test_parse_color_map(fixture_file: str, expected_grid: Grid[CardColor]):
    image_path = get_fixture_path(fixture_file)
    colors = parse_color_map(image_path=image_path)
    grid = Grid.from_list(row_size=GRID_WIDTH, items=colors)
    diff = expected_grid.diff(other=grid)
    print_diff(diff)
    assert not diff
