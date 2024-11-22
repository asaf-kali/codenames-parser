from dataclasses import dataclass
from typing import Iterable, Iterator, NamedTuple, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass
class Point(Sequence[int]):
    x: int
    y: int

    @property
    def tuple(self):
        return self.x, self.y

    def __getitem__(self, index):
        return self.tuple[index]

    def __iter__(self):
        return iter(self.tuple)

    def __len__(self):
        return len(self.tuple)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __sub__(self, other):
        return self.__add__(-other)

    def __str__(self):
        return f"({self.x:.0f}, {self.y:.0f})"

    def distance(self, other: "Point") -> float:
        return float(np.linalg.norm(self - other))


@dataclass
class Size(Sequence[int]):
    width: int
    height: int

    @property
    def tuple(self):
        return self.width, self.height

    def __getitem__(self, index):
        return self.tuple[index]

    def __iter__(self):
        return iter(self.tuple)

    def __len__(self):
        return len(self.tuple)

    def __str__(self):
        return f"({self.width}, {self.height})"


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.w, self.h))

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

    def center_distance(self, other: "Box") -> float:
        return float(np.linalg.norm(self.center - other.center))

    def overlaps(self, other: "Box") -> bool:
        return (
            self.x_min < other.x_max
            and self.x_max > other.x_min
            and self.y_min < other.y_max
            and self.y_max > other.y_min
        )

    def intersection(self, other: "Box") -> "Box":
        x_min = max(self.x_min, other.x_min)
        y_min = max(self.y_min, other.y_min)
        x_max = min(self.x_max, other.x_max)
        y_max = min(self.y_max, other.y_max)
        return Box(x=x_min, y=y_min, w=x_max - x_min, h=y_max - y_min)


@dataclass
class LetterBox(Box):
    letter: str


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
