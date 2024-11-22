from typing import Generator

import pytest

from codenames_parser.common.debug_util import set_save_debug_images
from codenames_parser.common.logging import configure_logging

configure_logging()

set_save_debug_images(enabled=False)


@pytest.fixture
def with_debug_images() -> Generator[None, None, None]:
    set_save_debug_images(enabled=True)
    yield None
    set_save_debug_images(enabled=False)
