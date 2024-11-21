import logging

import pytest
from codenames.generic.board import Board
from codenames.generic.card import Card

from codenames_parser.board.board_parser import parse_board
from codenames_parser.common.image_reader import read_image
from tests.b_board.cases import (
    BOARD1_CASE,
    BOARD1_TOP_CASE,
    BOARD2_CASE,
    BOARD2_TOP_CASE,
    BOARD3_TILT_CASE,
    BOARD3_TOP2_CASE,
    BOARD3_TOP_CASE,
    ParseBoardTestCase,
)
from tests.fixtures import get_fixture_path
from tests.utils import list_diff, print_diff

log = logging.getLogger(__name__)


def _test_parse_board(case: ParseBoardTestCase):
    image_path = get_fixture_path(f"boards/{case.language}/{case.fixture_file}")
    image = read_image(image_path)
    words = parse_board(image=image, language=case.language)
    cards = [Card(word=word, color=None) for word in words]
    board = Board(cards=cards, language=case.language)
    log.info(f"\n{board.as_table}")
    diff = list_diff(l1=case.expected_words, l2=words)
    print_diff(diff)
    assert len(diff) <= case.allowed_errors


def test_parse_board_case_1():
    _test_parse_board(BOARD1_CASE)


def test_parse_board_case_1_top():
    _test_parse_board(BOARD1_TOP_CASE)


def test_parse_board_case_2():
    _test_parse_board(BOARD2_CASE)


def test_parse_board_case_2_top():
    _test_parse_board(BOARD2_TOP_CASE)


@pytest.mark.skip("Tilt is not yet supported")
def test_parse_board_case_3_tilt():
    _test_parse_board(BOARD3_TILT_CASE)


def test_parse_board_case_3_top():
    _test_parse_board(BOARD3_TOP_CASE)


def test_parse_board_case_3_top2():
    _test_parse_board(BOARD3_TOP2_CASE)
