import typing

import cv2
import numpy as np

from examples.gtsrb.utils.domain.color.hsv_color import HSVColor
from examples.gtsrb.utils.domain.image.binary_mask import BinaryMask
from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.processing_steps.processing_step import ImageProcessingStep


class BinaryMaskFromHSVColor(ImageProcessingStep):

    def __init__(
            self,
            colors: typing.Tuple[HSVColor],
            name: typing.Optional[str] = None
    ):
        super().__init__(name)
        self._colors = colors

    def apply(self, fn_input: RGBImage) -> BinaryMask:
        mask = BinaryMask(
            np.zeros(fn_input.value.shape[0:2], dtype=np.uint8)
        )
        image_hsv = cv2.cvtColor(fn_input.value, cv2.COLOR_RGB2HSV)
        for color in self._colors:
            color_mask = cv2.inRange(image_hsv, color.lower_bound, color.upper_bound)
            mask.union(BinaryMask(color_mask))
        return mask
