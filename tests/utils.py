import logging

from codenames_parser.common.models import CellDiff

log = logging.getLogger(__name__)


def print_diff(diff: list[CellDiff]):
    if not diff:
        return
    log.error(f"Diff pretty print ({len(diff)} items):")
    for item in diff:
        log.error(item)
