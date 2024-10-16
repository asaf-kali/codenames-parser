from dataclasses import dataclass
from typing import Generic, Iterable, Iterator, NamedTuple, TypeVar

import numpy as np

T = TypeVar("T")


class Point(NamedTuple):
    x: int
    y: int

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __sub__(self, other):
        return self.__add__(-other)

    def __str__(self):
        return f"({self.x:.0f}, {self.y:.0f})"


class Box(NamedTuple):
    x: int
    y: int
    w: int
    h: int

    @property
    def x_min(self) -> int:
        return self.x

    @property
    def x_max(self) -> int:
        return self.x + self.w

    @property
    def y_min(self) -> int:
        return self.y

    @property
    def y_max(self) -> int:
        return self.y + self.h

    @property
    def x_center(self) -> int:
        return self.x + self.w // 2

    @property
    def y_center(self) -> int:
        return self.y + self.h // 2

    @property
    def center(self) -> Point:
        return Point(self.x_center, self.y_center)

    @property
    def area(self) -> int:
        return self.w * self.h


class P1P2(NamedTuple):
    p1: Point
    p2: Point


class Line(NamedTuple):
    rho: float  # distance from the origin
    theta: float  # angle in radians


@dataclass
class GridLines(Iterable[Line]):
    horizontal: list[Line]
    vertical: list[Line]

    @property
    def lines(self) -> list[Line]:
        return self.horizontal + self.vertical

    def __iter__(self) -> Iterator[Line]:  # type: ignore
        return iter(self.lines)


class Color(NamedTuple):
    b: int
    g: int
    r: int

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.b, self.g, self.r])

    def __str__(self) -> str:
        return f"({self.r},{self.g},{self.b})"


class CellDiff(NamedTuple):
    row: int
    col: int
    value_self: T  # type: ignore[valid-type]
    value_other: T  # type: ignore[valid-type]


class Grid(Generic[T]):
    def __init__(self, row_size: int):
        self.row_size = row_size
        self._rows: list[list[T]] = []

    @staticmethod
    def from_list(row_size: int, items: list[T]) -> "Grid[T]":
        grid: Grid[T] = Grid(row_size)
        for i in range(0, len(items), row_size):
            row = items[i : i + row_size]
            grid.append(row)
        return grid

    @staticmethod
    def from_rows(rows: list[list[T]]) -> "Grid[T]":
        if len(rows) == 0:
            raise ValueError("Rows cannot be empty")
        row_size = len(rows[0])
        grid: Grid[T] = Grid(row_size)
        for row in rows:
            grid.append(row)
        return grid

    @property
    def rows(self) -> list[list[T]]:
        return self._rows

    def append(self, row: list[T]) -> None:
        if len(row) != self.row_size:
            raise ValueError(f"Row size must be {self.row_size}")
        self._rows.append(row)

    def __iter__(self) -> Iterator[T]:
        for row in self._rows:
            yield from row

    def __getitem__(self, index: int) -> list[T]:
        return self._rows[index]

    def __len__(self) -> int:
        return len(self._rows) * self.row_size

    def __eq__(self, other):
        if not isinstance(other, Grid):
            return False
        if self.row_size != other.row_size:
            return False
        if len(self._rows) != len(other):
            return False
        diff = self.diff(other)
        return not diff

    def diff(self, other: "Grid[T]") -> list[CellDiff]:
        diff = []
        for i in range(len(self._rows)):
            for j in range(self.row_size):
                if self[i][j] != other[i][j]:
                    cell_diff = CellDiff(i, j, self[i][j], other[i][j])
                    diff.append(cell_diff)
        return diff
