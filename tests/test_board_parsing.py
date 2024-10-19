import logging
import os

import pytest

from codenames_parser.board.board_parser import parse_board
from codenames_parser.common.grid_detection import GRID_WIDTH
from codenames_parser.common.models import Grid
from tests.board_cases import BOARD_CASES
from tests.fixtures import get_fixture_path
from tests.utils import print_diff

log = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def set_env_vars():
    os.environ["DEBUG_DISABLED"] = "true"


@pytest.mark.parametrize("fixture_file,expected_words", BOARD_CASES)
def test_parse_board(fixture_file: str, expected_words: list[str]):
    image_path = get_fixture_path(fixture_file)
    colors = parse_board(image_path=image_path, language="heb")
    expected_grid = Grid.from_list(row_size=GRID_WIDTH, items=expected_words)
    actual_grid = Grid.from_list(row_size=GRID_WIDTH, items=colors)
    diff = expected_grid.diff(other=actual_grid)
    print_diff(diff)
    assert len(diff) < 10
