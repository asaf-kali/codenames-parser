import logging
import sys

FORMAT = "[%(asctime)s.%(msecs)03d] [%(levelname).4s] %(message)s"


def configure_logging():
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="%H:%M:%S", stream=sys.stdout)
