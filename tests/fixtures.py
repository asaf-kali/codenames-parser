import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
FIXTURES_DIR_PATH = os.path.join(DIR_PATH, "fixtures")


def get_fixture_path(*parts: str) -> str:
    return os.path.join(FIXTURES_DIR_PATH, *parts)
