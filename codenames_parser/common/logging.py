import logging
import sys


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname).4s] %(message)s", stream=sys.stdout)
