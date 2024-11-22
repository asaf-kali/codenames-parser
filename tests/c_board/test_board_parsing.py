import logging
from unittest import mock

import pytest
from codenames.generic.board import Board
from codenames.generic.card import Card

from codenames_parser.board.board_parser import parse_board
from codenames_parser.board.entrypoint import entrypoint
from codenames_parser.common.debug_util import set_run_id
from codenames_parser.common.image_reader import read_image
from codenames_parser.common.scale import downsample_image
from tests.c_board.cases import (
    BOARD1_CASE,
    BOARD1_TOP_CASE,
    BOARD1_TOP_SMALL_CASE,
    BOARD2_CASE,
    BOARD2_TOP_CASE,
    BOARD3_TILT_CASE,
    BOARD3_TOP2_CASE,
    BOARD3_TOP_CASE,
    BOARD4_TOP_CASE,
    ParseBoardTestCase,
)
from tests.fixtures import get_fixture_path
from tests.utils import list_diff, print_diff

log = logging.getLogger(__name__)


def _get_board_fixture_path(fixture_file: str, language: str) -> str:
    return get_fixture_path(f"boards/{language}/{fixture_file}")


def _test_parse_board(case: ParseBoardTestCase):
    image_path = _get_board_fixture_path(fixture_file=case.fixture_file, language=case.language)
    image = read_image(image_path)
    if case.downsample_factor != 1:
        image = downsample_image(image=image, factor=case.downsample_factor)
    words = parse_board(image=image, language=case.language)
    cards = [Card(word=word, color=None) for word in words]
    board = Board(cards=cards, language=case.language)
    log.info(f"\n{board.as_table}")
    diff = list_diff(l1=case.expected_words, l2=words)
    print_diff(diff)
    assert len(diff) <= case.allowed_errors
    if len(diff) < case.allowed_errors:
        log.warning(f"!!! Can be reduced !!! Allowed errors: {case.allowed_errors}, found: {len(diff)}")


@pytest.mark.slow
def test_parse_board_case_1():
    _test_parse_board(BOARD1_CASE)


@pytest.mark.slow
def test_parse_board_case_1_top():
    _test_parse_board(BOARD1_TOP_CASE)


def test_parse_board_case_1_top_small():
    _test_parse_board(BOARD1_TOP_SMALL_CASE)


@pytest.mark.slow
def test_parse_board_case_2():
    _test_parse_board(BOARD2_CASE)


@pytest.mark.slow
def test_parse_board_case_2_top():
    _test_parse_board(BOARD2_TOP_CASE)


@pytest.mark.slow
@pytest.mark.skip("Tilt is not yet supported")
def test_parse_board_case_3_tilt():
    _test_parse_board(BOARD3_TILT_CASE)


@pytest.mark.slow
def test_parse_board_case_3_top():
    _test_parse_board(BOARD3_TOP_CASE)


@pytest.mark.slow
def test_parse_board_case_3_top2():
    _test_parse_board(BOARD3_TOP2_CASE)


@pytest.mark.slow
def test_parse_board_case_4_top():
    _test_parse_board(BOARD4_TOP_CASE)


def test_entrypoint(with_debug_images: None):
    set_run_id(run_id="board_entrypoint")
    language = "heb"
    fixture_path = _get_board_fixture_path(fixture_file="board4_top.png", language=language)
    with mock.patch("sys.argv", ["", fixture_path, language]):
        words = entrypoint()
        assert len(words) == 25
