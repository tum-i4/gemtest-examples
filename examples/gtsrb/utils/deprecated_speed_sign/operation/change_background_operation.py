import typing

import cv2
import numpy as np

from examples.gtsrb.utils.deprecated_speed_sign.visual_tools.processing_step \
    .find_sign_by_border_color_sequence import find_sign_by_border_color
from examples.gtsrb.utils.deprecated_speed_sign.visual_tools.processing_step.validate_full_sign_mask import \
    ValidateFullSignMaskOptions
from examples.gtsrb.utils.domain.color.hsv_color import HSVColor
from examples.gtsrb.utils.domain.color.rgb_color import RGBColor
from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.processing_steps.processing_step import ImageProcessingStep


def _get_average_color(
        img: RGBImage,
        ignore_color_set: typing.Set[RGBColor]
) -> RGBColor:
    """Get average color from all pixels whose color is
      not included on the ignore list.

    Motivating use case of procedure is to combine it
    with a mask selection operation and skipping the black
    pixels i.e. unselected pixels.
    """
    avg_r, avg_g, avg_b = 0, 0, 0
    img_np_copy = img.value.copy()
    img_y, img_x = img_np_copy.shape[0:2]
    count = 0
    for i in range(img_y):
        for j in range(img_x):
            pixel_val = tuple(img_np_copy[i, j, :])
            if pixel_val not in ignore_color_set:
                avg_r += pixel_val[0]
                avg_g += pixel_val[1]
                avg_b += pixel_val[2]
                count += 1
    return avg_r // count, avg_g // count, avg_b // count


def _set_color(
        img: RGBImage,
        ignore_color_set: typing.Set[RGBColor],
        fill_color: RGBColor
) -> RGBImage:
    """Change color of all pixels whose color value
    is not on ignore list.

    Procedure returns a copy of the input image
    """
    img_np_copy = img.value.copy()
    img_y, img_x = img_np_copy.shape[0:2]
    for i in range(img_y):
        for j in range(img_x):
            pixel_val = tuple(img_np_copy[i, j, :])
            if pixel_val not in ignore_color_set:
                avg_r, avg_g, avg_b = fill_color
                img_np_copy[i, j, 0] = avg_r
                img_np_copy[i, j, 1] = avg_g
                img_np_copy[i, j, 2] = avg_b
    return RGBImage(img_np_copy)


class ChangeBackgroundOperation(ImageProcessingStep):
    """Attempts to confuse the tested model by changing toe background color
    to resemble the one on the border. The assumption is that the operation
    will make it difficult for the network to locate the sign.
    
    The operation is color agnostic, allowing to specify the color of the
    seeked border as [hsv_color.HSVColor] instance. Thus, this Operation
    should be reusable across a number of classes.

    The operation interpolates between the original background and a solid
    background colored in the border of the color. The weight of the color
    background i.e. the color intensity can be adjusted via [color_intensity].
    A value between 0.6 and 0.9 is advised.
    """

    def __init__(
            self,
            border_color: typing.Tuple[HSVColor],
            color_intensity: float = 0.6,
            name: typing.Optional[str] = None,
            full_sign_validation_options: ValidateFullSignMaskOptions =
            ValidateFullSignMaskOptions.default_options()
    ):
        assert 0 < color_intensity < 1, "Weight of the color overlay must be fractional"
        super(ChangeBackgroundOperation, self).__init__(name)
        self._border_color = border_color
        self._color_intensity = color_intensity
        self._full_sign_validation_options = full_sign_validation_options

    def apply(
            self, fn_input: RGBImage
    ) -> typing.Optional[RGBImage]:
        assert isinstance(fn_input, RGBImage)
        (
            border_mask_pipeline, background_mask_pipeline,
            full_sign_mask_pipeline
        ) = \
            find_sign_by_border_color(
                self._border_color,
                self._full_sign_validation_options
            )

        full_sign_mask = full_sign_mask_pipeline.apply(fn_input)

        if full_sign_mask is None:
            return None

        background_img = fn_input.get_crop(
            background_mask_pipeline.apply(fn_input)
        )
        border_img = fn_input.get_crop(
            border_mask_pipeline.apply(fn_input)
        )
        full_sign_img = fn_input.get_crop(full_sign_mask)
        not_black_set = {(0, 0, 0)}
        avg_border_color = _get_average_color(
            border_img, not_black_set
        )
        border_color_background = _set_color(
            background_img, not_black_set, avg_border_color
        )
        # Keep the texture of the background while superimposing
        # a layer of solid color equal to the border color.
        # A bias towards the color of the border is recommended.
        confusing_background: np.ndarray = cv2.addWeighted(
            border_color_background.value, self._color_intensity,
            background_img.value, (1 - self._color_intensity),
            0
        )
        return RGBImage(
            confusing_background + full_sign_img.value
        )
