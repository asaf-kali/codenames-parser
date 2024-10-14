import os

import pytest
from codenames.game.color import CardColor

from codenames_parser.main import main
from codenames_parser.models import Grid
from tests.cases import MAP_CASES
from tests.fixtures import get_fixture_path


@pytest.fixture(autouse=True)
def set_env_vars():
    os.environ["DEBUG_DISABLED"] = "true"


@pytest.mark.parametrize("fixture_file,expected_grid", MAP_CASES)
def test_map_parsing(fixture_file: str, expected_grid: Grid[CardColor]):
    image_path = get_fixture_path(fixture_file)
    result = main(image_path=image_path)
    diff = expected_grid.diff(other=result)
    assert not diff
