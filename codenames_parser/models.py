from typing import NamedTuple

import numpy as np


class Point(NamedTuple):
    x: int
    y: int


class P1P2(NamedTuple):
    p1: Point
    p2: Point


class Line(NamedTuple):
    rho: float  # distance from the origin
    theta: float  # angle in radians


class GridLines(NamedTuple):
    horizontal: list[Line]
    vertical: list[Line]


class Color(NamedTuple):
    b: int
    g: int
    r: int

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.b, self.g, self.r])

    def __str__(self) -> str:
        return f"({self.r},{self.g},{self.b})"
