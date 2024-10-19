import os

import pytest

from codenames_parser.common.logging import configure_logging

configure_logging()


@pytest.fixture(autouse=True)
def set_env_vars():
    os.environ["DEBUG_DISABLED"] = "true"
