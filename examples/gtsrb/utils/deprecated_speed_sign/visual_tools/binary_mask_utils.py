import math

import cv2
import numpy as np

from examples.gtsrb.utils.domain.point_2d import Point2D


def get_points_centroid(points: np.ndarray) -> Point2D:
    """Get coordinates of the centroid for the selected pixels.

    The selected/ unselected pixels follow the definition found
    in the BinaryMask class. Pass the inner value of the BinaryMask
    """
    points = points.copy()
    # Find centroid using the momentum technique
    m = cv2.moments(points, False)
    assert m['m00'] != 0, "No non-zero pixel values in image, you done goof"
    return Point2D(
        x=math.floor(m['m10'] / m['m00']),
        y=math.floor(m['m01'] / m['m00'])
    )
