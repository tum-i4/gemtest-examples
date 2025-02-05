import random

import cv2
import numpy as np

from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.processing_steps.processing_step import ImageProcessingStep, _Image


class Step1GeneratePolygonPoints(ImageProcessingStep):
    """Generates a polygon with either 3 or 4 points which is later used to overlay a shadow over an image.
    The polygon will always connect to at least 1 border of the image so that the shadow does not float in the image."""

    def __init__(self):
        super().__init__("shadow_polygon_generator")

    def apply(self, fn_input: _Image) -> np.ndarray:
        cols, rows, _ = fn_input.value.shape
        rows -= 1
        cols -= 1
        flag = random.choice(["triangle", "rectangle"])
        top_left_corner = [0, 0]
        bottom_left_corner = [0, random.randint(0, cols)]

        if flag == "triangle":
            # triangle
            top_right_corner = [random.randint(0, rows), 0]
            points = np.array([top_left_corner, top_right_corner, bottom_left_corner])
        else:
            # rectangle
            top_right_corner = [rows, 0]

            flag = random.choice(["bottom", "right"])
            bottom_right_corner = [rows, random.randint(0, cols)] if flag == "right" \
                else [random.randint(0, rows), cols]

            points = np.array([top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner])

        return points


class Step2FillPolygonWithShadow(ImageProcessingStep):
    """Takes a polygon and fills this area created by the polygon with a shadow.
    The effect of a shadow is created by decreasing the brightness in this are"""

    def __init__(self, points: np.ndarray, shade: float):
        self.points = points
        self.shade = shade
        super().__init__("apply_shadow_to_polygons")

    @staticmethod
    def _brightness_contrast_adjust_uint(img, alpha=1, beta=0, beta_by_max=False):
        dtype = np.dtype("uint8")

        max_value = 255

        lut = np.arange(0, max_value + 1).astype("float32")

        if alpha != 1:
            lut *= alpha
        if beta != 0:
            if beta_by_max:
                lut += beta * max_value
            else:
                lut += (alpha * beta) * np.mean(img)

        lut = np.clip(lut, 0, max_value).astype(dtype)
        img = cv2.LUT(img, lut)
        return img

    def apply(self, fn_input: _Image) -> _Image:
        image = fn_input.value
        mask = np.zeros((image.shape[:2]), image.dtype)
        mask = cv2.fillPoly(mask, [self.points], 1)
        flip = random.choice(["none", 0, 1, -1])
        if flip != "none":
            mask = cv2.flip(mask, flip)
        image_fg = cv2.bitwise_and(image, image, mask=mask)
        image_bg = image - image_fg
        image_fg = self._brightness_contrast_adjust_uint(image_fg, alpha=1, beta=-1 + self.shade)
        image_rgb = image_fg + image_bg
        return RGBImage(image_rgb)


class ShadowGenerationPipeline(ImageProcessingStep):
    """Takes an image and projects a shadow onto it."""

    def __init__(self, shade: float):
        self.shade = shade
        super().__init__("shadow_generator")

    def apply(self, fn_input: _Image) -> _Image:
        points = Step1GeneratePolygonPoints().apply(fn_input)
        image_rgb = Step2FillPolygonWithShadow(points, self.shade).apply(fn_input)
        return image_rgb
