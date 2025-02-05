import typing

import cv2
import numpy as np

from examples.gtsrb.utils.domain import point_2d
from examples.gtsrb.utils.domain.image.binary_mask import BinaryMask

PerspectiveTransformFn = typing.Callable[[np.ndarray], np.ndarray]


def _get_limit_coordinates(points: np.ndarray) -> typing.Iterable[point_2d.Point2D]:
    """Get points with extreme coordinates i.e. points that have
    X_leftmost, X_rightmost, Y_top, Y_bot.

    If multiple points match the predicate, only one is returned. Points are
    returned in counter-clockwise order starting from top.
    """
    top_y, bot_y, left_x, right_x = [
        point_2d.Point2D(y=float('inf'), x=None),
        point_2d.Point2D(y=float('-inf'), x=None),
        point_2d.Point2D(y=None, x=float('inf')),
        point_2d.Point2D(y=None, x=float('-inf'))
    ]
    _y, _x = points.shape
    for dy in range(_y):
        for dx in range(_x):
            if points[dy][dx] == BinaryMask.SELECTED_PIXEL_VALUE:
                if dy < top_y.y:
                    top_y.y = dy
                    top_y.x = dx
                if dy > bot_y.y:
                    bot_y.x = dx
                    bot_y.y = dy
                if dx < left_x.x:
                    left_x.x = dx
                    left_x.y = dy
                if dx > right_x.x:
                    right_x.x = dx
                    right_x.y = dy
    return [top_y, left_x, bot_y, right_x]


def calculate_perspective_distortion(
        mask: BinaryMask, control_mask: BinaryMask
) -> typing.Tuple[PerspectiveTransformFn, PerspectiveTransformFn]:
    """Returns two Callables that allow switching to and from the control perspective
    found in control_mask.

    Returns: Tuple containing: [
        Transform BinaryMask perspective to control perspective callable
        Reverse BinaryMask from control perspective callable
    ]
    """
    coords = [(p.x, p.y) for p in _get_limit_coordinates(mask.value)]
    coords_control = [(p.x, p.y) for p in _get_limit_coordinates(control_mask.value)]

    perspective_transform_matrix = cv2.getPerspectiveTransform(
        np.float32(coords), np.float32(coords_control)
    )
    inverse_perspective_transform_matrix = cv2.getPerspectiveTransform(
        np.float32(coords_control), np.float32(coords)
    )

    d_size = mask.value.shape

    def op(img: np.ndarray) -> np.ndarray:
        np_arr = cv2.warpPerspective(
            img, perspective_transform_matrix, d_size[::-1]
        )
        return np_arr

    def reverse_op(img: np.ndarray) -> np.ndarray:
        np_arr = cv2.warpPerspective(
            img, inverse_perspective_transform_matrix, d_size[::-1]
        )
        return np_arr

    return op, reverse_op
