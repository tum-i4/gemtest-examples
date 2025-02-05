import numpy as np

from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.domain.image.rgba_image import RGBAImage
from examples.gtsrb.utils.processing_steps.processing_step \
    import ImageProcessingStep


class Overlay(ImageProcessingStep):
    def __init__(self, foreground: RGBAImage, x_offset: int = 0, y_offset: int = 0):
        super().__init__()
        self.foreground: np.ndarray = foreground.value
        self.x_offset = x_offset
        self.y_offset = y_offset

    def apply(self, fn_input: RGBImage) -> RGBImage:
        """Overlay foreground onto fn_input taking into account alpha of the foreground"""
        background = np.copy(fn_input.value)
        bg_h, bg_w, bg_channels = background.shape
        fg_h, fg_w, fg_channels = self.foreground.shape

        assert background is not None
        assert bg_channels == 3, f'background image should have exactly 3 channels ' \
                                 f'(rgb_image.RGBImage). Found:{bg_channels}'
        assert fg_channels == 4, f'foreground image should have exactly 4 channels ' \
                                 f'(RGBAImage). Found:{fg_channels}'

        w = min(fg_w, bg_w, fg_w + self.x_offset, bg_w - self.x_offset)
        h = min(fg_h, bg_h, fg_h + self.y_offset, bg_h - self.y_offset)

        if w < 1 or h < 1:
            return RGBImage(background)

        # clip foreground and background images to the overlapping regions
        bg_x = max(0, self.x_offset)
        bg_y = max(0, self.y_offset)
        fg_x = max(0, self.x_offset * -1)
        fg_y = max(0, self.y_offset * -1)
        foreground = self.foreground[fg_y:fg_y + h, fg_x:fg_x + w]
        bg_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

        # separate alpha and color channels from the foreground image
        foreground_colors = foreground[:, :, :3]
        alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

        # construct an alpha_mask that matches the image shape
        alpha_mask = alpha_channel[:, :, np.newaxis]

        # combine the background with the overlay image weighted by alpha
        composite = bg_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

        # overwrite the section of the background image that has been updated
        background[bg_y:bg_y + h, bg_x:bg_x + w] = composite
        return RGBImage(background)
