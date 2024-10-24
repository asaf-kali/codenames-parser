import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ListDiff:
    index: int
    value_self: Any
    value_other: Any

    def __str__(self):
        return f"{self.index}: [{self.value_self}] != [{self.value_other}]"


def list_diff(l1: list, l2: list) -> list[ListDiff]:
    diffs = []
    if len(l1) != len(l2):
        raise ValueError(f"Lists are not of the same size: {len(l1)} != {len(l2)}")
    for i, (v1, v2) in enumerate(zip(l1, l2)):
        if v1 != v2:
            diff = ListDiff(index=i, value_self=v1, value_other=v2)
            diffs.append(diff)
    return diffs


def print_diff(diff: list[ListDiff]):
    if not diff:
        log.info("No diff found!")
        return
    log.error(f"Diff pretty print ({len(diff)} items):")
    for item in diff:
        log.error(item)
