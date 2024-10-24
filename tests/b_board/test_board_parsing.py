import logging

from codenames.game.board import Board
from codenames.game.card import Card

from codenames_parser.board.board_parser import parse_board
from codenames_parser.common.grid_detection import GRID_WIDTH
from codenames_parser.common.models import Grid
from tests.b_board.cases import BOARD1_TOP_CASE, BOARD3_TOP_CASE, ParseBoardTestCase
from tests.fixtures import get_fixture_path
from tests.utils import print_diff

log = logging.getLogger(__name__)


def _test_parse_board(case: ParseBoardTestCase):
    image_path = get_fixture_path(case.fixture_file)
    words = parse_board(image_path=image_path, language=case.language)
    cards = [Card(word=word) for word in words]
    board = Board(cards=cards, language=case.language)
    log.info(f"\n{board.as_table}")
    expected_grid = Grid.from_list(row_size=GRID_WIDTH, items=case.expected_words)
    actual_grid = Grid.from_list(row_size=GRID_WIDTH, items=words)
    diff = expected_grid.diff(other=actual_grid)
    print_diff(diff)
    assert len(diff) <= 1


def test_parse_board_case_1_top():
    _test_parse_board(BOARD1_TOP_CASE)


def test_parse_board_case_3_top():
    _test_parse_board(BOARD3_TOP_CASE)
