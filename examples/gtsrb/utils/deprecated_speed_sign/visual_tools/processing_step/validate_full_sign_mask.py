import collections
import dataclasses
import math
import typing

import numpy as np

from examples.gtsrb.utils.deprecated_speed_sign.visual_tools.binary_mask_utils import get_points_centroid
from examples.gtsrb.utils.domain import point_2d
from examples.gtsrb.utils.domain.image.binary_mask import BinaryMask
from examples.gtsrb.utils.processing_steps.processing_step import ImageProcessingStep


@dataclasses.dataclass
class ValidateFullSignMaskOptions:

    def __init__(self, min_area_selected_pipeline: float, max_area_selected_pipeline: float,
                 tolerated_displacement_from_center: float,
                 search_range_gaps_inside_center: float):
        self.min_area_selected_pipeline = min_area_selected_pipeline
        self.max_area_selected_pipeline = max_area_selected_pipeline
        self.tolerated_displacement_from_center = tolerated_displacement_from_center
        self.search_range_gaps_inside_center = search_range_gaps_inside_center

    @classmethod
    def default_options(cls) -> 'ValidateFullSignMaskOptions':
        return ValidateFullSignMaskOptions(
            min_area_selected_pipeline=0.2,
            max_area_selected_pipeline=0.8,
            tolerated_displacement_from_center=20,
            search_range_gaps_inside_center=20,
        )


class ValidateFullSignMask(ImageProcessingStep):
    """Validates that the lookup by border lookup worked.

    The processing step implements three heuristics:
        1. The mask area occupies a significant area of the image
        2. The center of the selected points is not too far from the center
            of the image
        3. The area around the center of the selected points does not contain
            empty pixels.
    These options can be controlled through the passed ValidateFullSignMaskOptions
    object. [min_area] and [max_area] control the first heuristic, [tolerated_dist]
    controls the second heuristic and [center_range] controls the final
    heuristic.
    """

    def __init__(
            self,
            options: ValidateFullSignMaskOptions,
            name: typing.Optional[str] = None
    ):
        super().__init__(name=name)
        self._min_area = options.min_area_selected_pipeline
        self._max_area = options.max_area_selected_pipeline
        self._tolerated_dist = options.tolerated_displacement_from_center
        self._search_range_gaps = options.search_range_gaps_inside_center

    def apply(
            self, fn_input: BinaryMask
    ) -> typing.Optional[BinaryMask]:
        np_copy: np.ndarray = fn_input.value.copy()
        area_counter = collections.Counter(list(np_copy.flatten()))
        # Assert that the selected area is between
        # (empirically measured to work) thresholds
        white_p = BinaryMask.SELECTED_PIXEL_VALUE
        dark_p = BinaryMask.UNSELECTED_PIXEL_VALUE
        mask_area = area_counter[white_p] / (
                area_counter[white_p] + area_counter[dark_p]
        )
        if mask_area < self._min_area or mask_area > self._max_area:
            return None
        # Check that the centroid of the selected points does not deviate
        # too much from the true center of the image. This heuristic
        # is built on the fact that all images in the dataset.py have the road
        # sign in the center
        centroid: point_2d.Point2D = get_points_centroid(
            fn_input.value
        )
        true_centroid_y, true_centroid_x = (
            np_copy.shape[0] // 2, np_copy.shape[1] // 2
        )
        euclid_dist = math.sqrt(
            (centroid.y - true_centroid_y) ** 2 + (centroid.x - true_centroid_x) ** 2
        )
        average_img_side = (np_copy.shape[0] + np_copy.shape[1]) / 2
        if euclid_dist > self._tolerated_dist * average_img_side:
            return None
        # Check that are is no middle gap e.g. blue sign white arrow
        # was improperly picked from the background
        for y in range(int(-self._search_range_gaps * average_img_side),
                       int(self._search_range_gaps * average_img_side) + 1):
            for x in range(int(-self._search_range_gaps * average_img_side),
                           int(self._search_range_gaps * average_img_side) + 1):
                if np_copy[centroid.y + y][centroid.x + x] != white_p:
                    return None
        # Image is assumed to pass the quality check
        return fn_input
