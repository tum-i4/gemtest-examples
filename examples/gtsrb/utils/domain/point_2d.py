import dataclasses
import typing
from typing import Optional

# Possibly null numerical value
Coordinate2D = Optional[typing.Union[float, int]]


@dataclasses.dataclass
class Point2D:
    """A point in 2D plane.

    In-framework, the point is used for various geometric computations
    done on image binary masks under the [visual_tools] module. The
    main point of the class is to enforce code readability ("is this
    API using the <X, Y> or <Y, X> convention?").
    """
    x: Coordinate2D
    y: Coordinate2D

    def __init__(self, y: Coordinate2D, x: Coordinate2D):
        self.y = y
        self.x = x
