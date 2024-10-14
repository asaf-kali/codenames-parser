import os

import pytest

from codenames_parser.main import main
from tests.cases import CASE_CUT_ROTATED, MapTestCase
from tests.fixtures import get_fixture_path


@pytest.fixture(autouse=True)
def set_env_vars():
    os.environ["DEBUG_DISABLED"] = "true"


@pytest.mark.parametrize("map_test_case", [CASE_CUT_ROTATED])
def test_map_parsing(map_test_case: MapTestCase):
    image_path = get_fixture_path(map_test_case.fixture_file)
    result = main(image_path=image_path)
    assert result == map_test_case.expected_grid
