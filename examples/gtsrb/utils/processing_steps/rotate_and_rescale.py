from typing import Optional

import cv2
import numpy as np

from examples.gtsrb.utils.processing_steps.processing_step import ImageProcessingStep, _Image


class RotateAndRescale(ImageProcessingStep):
    def __init__(self,
                 angle_in_degrees: float = 0.0,
                 scale: float = 1.0,
                 # from superclass
                 name: Optional[str] = None
                 ):
        """Rotate and rescale an image. angle_in_degrees is positive for clockwise rotation"""
        if name is None:
            name = f'rotate_and_rescale_{angle_in_degrees}_degrees_' \
                   f'scale_{scale}'
        super().__init__(name)
        self.angle_in_degrees = angle_in_degrees
        self.scale = scale

    def apply(self, fn_input: _Image) -> _Image:
        """Rescale and rotate an image. angle_in_degrees is positive for clockwise rotation"""
        img = fn_input.value
        (h, w) = np.array(img.shape[:2])
        center = (w / 2, h / 2)

        rot_matrix = cv2.getRotationMatrix2D(center, -self.angle_in_degrees, self.scale)
        rotated_img = cv2.warpAffine(img, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)
        if img.shape[-1] == 1:
            rotated_img = rotated_img[..., np.newaxis]
        # assert the image dimensions and print dimensions if not correct
        assert fn_input.value.shape == rotated_img.shape, \
            f"Assertion Error in RotateAndRescale: Got input={fn_input.value.shape} " \
            f"and produced output={rotated_img.shape} with type {type(fn_input)(rotated_img)} "

        # Return an object of the same type as fn_input
        return type(fn_input)(rotated_img)
