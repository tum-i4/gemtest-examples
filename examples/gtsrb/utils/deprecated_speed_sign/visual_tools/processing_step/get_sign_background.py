import typing

import cv2

from examples.gtsrb.utils.deprecated_speed_sign.visual_tools.processing_step.fill_unselected_values import \
    FillUnselectedValues
from examples.gtsrb.utils.domain import point_2d
from examples.gtsrb.utils.domain.image.binary_mask import BinaryMask
from examples.gtsrb.utils.processing_steps.processing_step import ImageProcessingStep


class GetSignBackground(ImageProcessingStep):
    """Get binary mask that matches all pixels outside the sign.

    Function starts a fill algorithm from the image corners, filling all
    points outside the border. The strict background is obtained through XOR.
    """

    def apply(
            self, fn_input: BinaryMask
    ) -> typing.Optional[BinaryMask]:
        border_points_np_array = fn_input.value
        image_corners = [
            point_2d.Point2D(y=0, x=border_points_np_array.shape[0] - 1),
            point_2d.Point2D(y=border_points_np_array.shape[0] - 1, x=0),
            point_2d.Point2D(y=border_points_np_array.shape[0] - 1,
                             x=border_points_np_array.shape[1] - 1),
            point_2d.Point2D(y=0, x=border_points_np_array.shape[1] - 1)
        ]

        # This fill will select all points outside the previously identified border
        background_plus_border_mask = FillUnselectedValues(
            origin_points=image_corners).apply(fn_input)
        # Separate border from background
        background_mask = cv2.bitwise_xor(
            background_plus_border_mask.value, border_points_np_array
        )
        return BinaryMask(background_mask)
