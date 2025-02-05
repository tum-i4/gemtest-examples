from typing import Optional

import numpy as np

from examples.gtsrb.utils.domain.image.binary_mask import BinaryMask
from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.domain.image.rgba_image import RGBAImage
from examples.gtsrb.utils.processing_steps.overlay import Overlay
from examples.gtsrb.utils.processing_steps.processing_step import ImageProcessingStep
from examples.gtsrb.utils.processing_steps.rotate_and_rescale import \
    RotateAndRescale


class RotateOperation(ImageProcessingStep):
    def __init__(self,
                 angle_in_degrees: float = 0.0,
                 scale: float = 1.0,
                 # from superclass
                 name: Optional[str] = None
                 ):
        """Rotate and rescale a sign while keeping the background intact. angle_in_degrees
        is positive for clockwise rotation"""
        if name is None:
            name = f'rotate_{angle_in_degrees}_degrees_' \
                   f'scale_{scale}'
        super().__init__(name)
        self._scale = scale

        # s = np.sin(np.deg2rad(angle_in_degrees % 90)) \
        #   + np.cos(np.deg2rad(angle_in_degrees % 90))

        # self._rotation = RotateAndRescale(angle_in_degrees, scale / s)
        self._rotation = RotateAndRescale(angle_in_degrees, 1.0)

    def apply(self, fn_input: RGBImage) -> RGBImage:
        background = fn_input.copy()
        shape = list(fn_input.value.shape)
        shape[2] = 1
        mask = BinaryMask(np.ones(shape) * 255)

        rotated_input = self._rotation.apply(fn_input)
        rotated_mask = self._rotation.apply(mask)

        rotated_mask_arr = rotated_mask.value[..., np.newaxis]
        rgba = np.append(rotated_input.value, rotated_mask_arr, axis=2)
        foreground = RGBAImage(rgba)
        x_offset = int(shape[0] * (1 - self._scale) / 2)
        y_offset = int(shape[1] * (1 - self._scale) / 2)
        return Overlay(foreground, x_offset=x_offset, y_offset=y_offset).apply(background)


if __name__ == '__main__':
    ''' Demonstrate overlaying of different images on a 'Forbidden' sign '''
    # OVERLAY_IMAGES_DIR = 'examples/gtsrb/assets/overlay/buildings'
    image = RGBImage.read_from_disk('examples/gtsrb/assets/forbidden.png')
    RotateOperation(angle_in_degrees=30).apply(image).plot()
