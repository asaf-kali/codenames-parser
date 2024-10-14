from typing import Generic, Iterable, NamedTuple, TypeVar

import numpy as np

T = TypeVar("T")


class Point(NamedTuple):
    x: int
    y: int


class Box(NamedTuple):
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h


class P1P2(NamedTuple):
    p1: Point
    p2: Point


class Line(NamedTuple):
    rho: float  # distance from the origin
    theta: float  # angle in radians


class GridLines(NamedTuple):
    horizontal: list[Line]
    vertical: list[Line]

    def __iter__(self) -> Iterable[Line]:
        return iter(self.horizontal + self.vertical)


class Color(NamedTuple):
    b: int
    g: int
    r: int

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.b, self.g, self.r])

    def __str__(self) -> str:
        return f"({self.r},{self.g},{self.b})"


class Grid(Generic[T]):
    def __init__(self, row_size: int):
        self.row_size = row_size
        self._rows = []

    @staticmethod
    def from_rows(rows: list[list[T]]) -> "Grid[T]":
        if len(rows) == 0:
            raise ValueError("Rows cannot be empty")
        row_size = len(rows[0])
        grid = Grid(row_size)
        for row in rows:
            grid.append(row)
        return grid

    def append(self, row: list[T]) -> None:
        if len(row) != self.row_size:
            raise ValueError(f"Row size must be {self.row_size}")
        self._rows.append(row)

    def __getitem__(self, index: int) -> list[T]:
        return self._rows[index]

    def __len__(self) -> int:
        return len(self._rows)

    def __eq__(self, other):
        if not isinstance(other, Grid):
            return False
        if self.row_size != other.row_size:
            return False
        row_count = len(self._rows)
        if row_count != len(other):
            return False
        for i in range(row_count):
            for j in range(self.row_size):
                if self[i][j] != other[i][j]:
                    return False
        return True
