import copy
import typing

import numpy as np

from examples.gtsrb.utils.domain import point_2d
from examples.gtsrb.utils.domain.image.binary_mask import BinaryMask
from examples.gtsrb.utils.processing_steps.processing_step import ImageProcessingStep


class FillUnselectedValues(ImageProcessingStep):
    """Transform all unselected pixels starting from origin points into selected.

    Flexible FILL algorithm that allows multiple start points and custom empty
    and fill in values.

    The algorithm allows for multiple fill start points and custom FILL and EMPTY
    pixel values.
    """

    def __init__(
            self,
            origin_points: typing.List[point_2d.Point2D],
            fill_value: int = BinaryMask.SELECTED_PIXEL_VALUE,
            empty_value: int = BinaryMask.UNSELECTED_PIXEL_VALUE,
            name: typing.Optional[str] = None
    ):
        super(FillUnselectedValues, self).__init__(name)
        self._origin_points = origin_points
        self._fill_value = fill_value
        self._empty_value = empty_value

    def apply(
            self, fn_input: BinaryMask
    ) -> typing.Optional[BinaryMask]:
        points_c: np.ndarray = fn_input.value.copy()
        y_upp_bound, x_upp_bound = points_c.shape
        directions = [
            point_2d.Point2D(x=0, y=-1),  # UP
            point_2d.Point2D(x=1, y=0),  # RIGHT
            point_2d.Point2D(x=0, y=1),  # DOWN
            point_2d.Point2D(x=-1, y=0)  # LEFT
        ]
        queue: typing.List[point_2d.Point2D] = copy.deepcopy(self._origin_points)
        while len(queue) != 0:
            p = queue.pop()
            x, y = p.x, p.y
            for direction in directions:
                xx, yy = x + direction.x, y + direction.y
                if (
                        yy >= y_upp_bound or xx >= x_upp_bound
                        or yy < 0 or xx < 0
                ):
                    # Out of boundaries
                    continue
                if points_c[yy][xx] == self._empty_value:
                    points_c[yy][xx] = self._fill_value
                    queue.append(point_2d.Point2D(x=xx, y=yy))
        return BinaryMask(points_c)
