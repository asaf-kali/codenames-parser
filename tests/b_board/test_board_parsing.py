import logging

import pytest
from codenames.game.board import Board
from codenames.game.card import Card

from codenames_parser.board.board_parser import parse_board
from codenames_parser.common.grid_detection import GRID_WIDTH
from codenames_parser.common.models import Grid
from tests.b_board.cases import BOARD_CASES
from tests.fixtures import get_fixture_path
from tests.utils import print_diff

log = logging.getLogger(__name__)


@pytest.mark.parametrize("fixture_file,language,expected_words", BOARD_CASES)
def test_parse_board(fixture_file: str, language: str, expected_words: list[str]):
    image_path = get_fixture_path(fixture_file)
    words = parse_board(image_path=image_path, language=language)
    cards = [Card(word=word) for word in words]
    board = Board(cards=cards, language=language)
    log.info(f"\n{board.as_table}")
    expected_grid = Grid.from_list(row_size=GRID_WIDTH, items=expected_words)
    actual_grid = Grid.from_list(row_size=GRID_WIDTH, items=words)
    diff = expected_grid.diff(other=actual_grid)
    print_diff(diff)
    assert len(diff) < 10
