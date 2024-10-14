import logging
import os

import pytest
from codenames.game.color import CardColor

from codenames_parser.main import main
from codenames_parser.models import CellDiff, Grid
from tests.cases import MAP_CASES
from tests.fixtures import get_fixture_path

log = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def set_env_vars():
    os.environ["DEBUG_DISABLED"] = "true"


@pytest.mark.parametrize("fixture_file,expected_grid", MAP_CASES)
def test_map_parsing(fixture_file: str, expected_grid: Grid[CardColor]):
    image_path = get_fixture_path(fixture_file)
    result = main(image_path=image_path)
    diff = expected_grid.diff(other=result)
    _print_diff(diff)
    assert not diff


def _print_diff(diff: list[CellDiff]):
    if not diff:
        return
    log.error("Diff pretty print:")
    for item in diff:
        log.error(item)
