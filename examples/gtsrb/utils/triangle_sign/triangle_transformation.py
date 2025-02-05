from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from examples.gtsrb.utils.domain.color.hsv_color import COLOR_RANGES, HSVColor
from examples.gtsrb.utils.domain.image.binary_mask import BinaryMask
from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.processing_steps.binary_mask_from_hsv_color import \
    BinaryMaskFromHSVColor
from examples.gtsrb.utils.processing_steps.processing_step import ImageProcessingStep, \
    _Image, ProcessingStep
from examples.gtsrb.utils.triangle_sign.triangle_error import TriangleDetectionError
from examples.gtsrb.utils.triangle_sign.triangle_preprocessing import TriangleCornerExtractor, \
    BoxFilter, \
    ExtractBorderPixels, AdaptedBorderMask, FindTriangleContour, ImageFromContour, TriangleCenterCheck


class TriangleDetector(ProcessingStep):
    """Applies a color mask to the image and then tries to detect a triangle in the image"""

    def __init__(self, colors: Tuple[HSVColor]):
        self.colors = colors
        name = f'triangle_detector_color_mask_degrees_'
        super().__init__(name)

    def apply(self, fn_input: RGBImage) -> Tuple[RGBImage, np.ndarray]:
        color_mask: BinaryMask = BinaryMaskFromHSVColor(self.colors).apply(fn_input)
        outer_contour = FindTriangleContour().apply(color_mask)
        filtered_img = ImageFromContour(outer_contour).apply(fn_input)
        return filtered_img, outer_contour


@dataclass
class DeconstructedTriangleSign:
    """A data class which contains all necessary parts of a triangular sign needed for morphing"""
    full_img: _Image
    outer_triangle: _Image
    inner_triangle: _Image
    centroid: Tuple[int, int]


class TriangleExtractor(ProcessingStep):
    """Extracts the outer (red + white part) and inner (white part) of a triangular sign.
    Also detects the corner points of the inner triangle"""

    def __init__(self):
        name = "triangle_sign_extractor"
        super().__init__(name)

    def apply(self, fn_input: _Image) -> DeconstructedTriangleSign:
        outer_triangle, outer_contour = TriangleDetector(COLOR_RANGES.RED_COLOR.value).apply(fn_input)
        inner_triangle, inner_contour = TriangleDetector(COLOR_RANGES.WHITE_COLOR.value).apply(outer_triangle)

        centroid = TriangleCenterCheck().apply(inner_contour, outer_contour)
        return DeconstructedTriangleSign(fn_input, outer_triangle, inner_triangle, centroid)


class CheckCornersAndCentroid(ProcessingStep):
    """Compares the distance between a provided centroid and an array of corners. If the distance is bigger than the
    threshold then False is returned otherwise True."""

    def __init__(self, threshold=3):
        self.threshold = threshold
        super().__init__()

    def apply(self, centroid: Tuple[int, int], corners: np.ndarray) -> bool:
        corner_centroid = [0, 0]
        for corner in corners:
            corner_centroid[0] += corner[0]
            corner_centroid[1] += corner[1]
        corner_centroid[0] = corner_centroid[0] / 3
        corner_centroid[1] = corner_centroid[1] / 3
        dx = (corner_centroid[0] - centroid[0]) ** 2
        dy = (corner_centroid[1] - centroid[1]) ** 2
        distance = np.sqrt(dx + dy)
        if distance > self.threshold:
            return False
        else:
            return True


class WarpTriangle(ImageProcessingStep):
    """ Does an affine transformation of a triangle. The goal is to bring one triangle into the same form, size and
    orientation as a provided triangle from another image. Outputs the warped imaged of the provided triangle"""

    def __init__(self, fg_tri_corners: np.ndarray, bg_tri_corners: np.ndarray,
                 output_shape: Tuple[int, int]):
        super().__init__("triangle_affine_warp")
        self.fg_tri_corners = fg_tri_corners
        self.bg_tri_corners = bg_tri_corners
        self.output_shape = output_shape

    def apply(self, foreground_img: RGBImage) -> RGBImage:
        M = cv2.getAffineTransform(self.fg_tri_corners, self.bg_tri_corners)
        rows, cols = self.output_shape
        warped_fg = RGBImage(cv2.warpAffine(foreground_img.value, M, (cols, rows)))
        return warped_fg


class TriangleOptimizationPipeline(ImageProcessingStep):
    """Used for smoothing out the edges of 2 combined triangle signs."""

    def __init__(self, background_img: RGBImage):
        super().__init__()
        self.bg_img = background_img

    def apply(self, warped_fg: RGBImage) -> RGBImage:
        border_mask = AdaptedBorderMask(invert=True).apply(warped_fg)
        full_image_outer = RGBImage(cv2.bitwise_and(self.bg_img.value, self.bg_img.value, mask=border_mask.value))
        fg_border_pixels = ExtractBorderPixels().apply(warped_fg)
        bg_border_pixels = ExtractBorderPixels().apply(full_image_outer)
        pixel_list = list(np.unique(fg_border_pixels + bg_border_pixels, axis=0))

        final_triangle = full_image_outer.value + warped_fg.value
        final_triangle = BoxFilter(pixel_list).apply(RGBImage(final_triangle))
        return final_triangle


class TriangleMorph(ImageProcessingStep):
    """Takes two images of triangle signs and cuts out the white inner triangle of the second image and inserts
    the triangle into the inner white triangle of the first sign."""

    def __init__(self, background_img: _Image):
        self.background_image = background_img
        super().__init__()

    def apply(self, fn_input: _Image) -> RGBImage:
        bg_img = self.background_image
        bg_tri = TriangleExtractor().apply(bg_img)
        fg_tri = TriangleExtractor().apply(fn_input)

        bg_shape = bg_img.value.shape[:2]
        fg_corners = TriangleCornerExtractor().apply(fg_tri.inner_triangle)
        bg_corners = TriangleCornerExtractor().apply(bg_tri.inner_triangle)

        centroid_check1 = CheckCornersAndCentroid().apply(fg_tri.centroid, fg_corners)
        centroid_check2 = CheckCornersAndCentroid().apply(bg_tri.centroid, bg_corners)
        if centroid_check1 and centroid_check2:
            warped_fg = WarpTriangle(fg_corners, bg_corners, bg_shape).apply(fg_tri.inner_triangle)
            final_triangle = TriangleOptimizationPipeline(bg_img).apply(warped_fg)
            return final_triangle
        raise TriangleDetectionError("Centroids don't match")
