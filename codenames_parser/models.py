from typing import NamedTuple


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
