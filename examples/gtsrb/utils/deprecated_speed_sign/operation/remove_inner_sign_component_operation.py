from typing import Optional, Tuple

import cv2
import numpy as np

from examples.gtsrb.utils.deprecated_speed_sign.visual_tools.processing_step.fill_unselected_values import \
    FillUnselectedValues
from examples.gtsrb.utils.domain import point_2d
from examples.gtsrb.utils.domain.image.binary_mask import BinaryMask
from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.processing_steps.processing_step import ImageProcessingStep
from examples.gtsrb.utils.processing_steps.processing_step_sequence import ProcessingStepSequence

_BinaryMaskOrNone = Optional[BinaryMask]
_SELECTED_PIXEL = BinaryMask.SELECTED_PIXEL_VALUE
_UNSELECTED_PIXEL = BinaryMask.UNSELECTED_PIXEL_VALUE


def _get_background_patch(
        white_mask: np.ndarray, seeked_size: int = 7
) -> Optional[Tuple[int, int]]:
    """Patch used to cover up the component of interest."""
    size = white_mask.shape[0]
    for i in range(size):
        for j in range(size):
            if white_mask[i, j] == _SELECTED_PIXEL:
                found = True
                for ii in range(-seeked_size, seeked_size + 1):
                    for jj in range(-seeked_size, seeked_size + 1):
                        if white_mask[i + ii, j + jj] != _SELECTED_PIXEL:
                            found = False
                if found:
                    return i, j
    # No patch of requested size could be found
    # Fallback to searching for a smaller patch
    return None


def _cover_component_pixels(
        component_mask: np.ndarray, white_pixels_mask: np.ndarray,
        target_rgb_image: np.ndarray, patch_size: int = 3
) -> Optional[np.ndarray]:
    """Covers all selected pixels from [component_mask] with a patch sampled
    from pixels selected in [white_pixels_mask]. Function returns a copy
    of [target_rgb_image]. [patch_size] can be adjusted to be different
    from [seeked_size] from [_get_background_patch] above. The intention
    is to avoid border pixels that belong to the red border but are too
    light to be classified under it, thus ending up in background white
    pixels. A way too avoid this is to set [_get_background_patch] [seeked_size]
    to a high value in order to select a more centric patch, than set [patch_size]
    to a small value in order to apply only the most centric pixels.
    """
    size = target_rgb_image.shape[0]
    rgb_image_copy = target_rgb_image.copy()
    component_mask_copy = component_mask.copy()
    # wi, wj is the center of the patch
    result = _get_background_patch(white_pixels_mask)
    if result is None:
        return None
    wi, wj = result
    # i, j iterate over the pixels of [target_rgb_image]
    for i in range(size):
        for j in range(size):
            if component_mask_copy[i, j] == _SELECTED_PIXEL:
                # Cover this pixel and the surrounding area
                # ii, jj iterate over the patch
                for ii in range(-patch_size, patch_size + 1):
                    for jj in range(-patch_size, patch_size + 1):
                        # Copy the values identified in the patch
                        # over the pixels selected in [component_mask]
                        # We copy over pixels around a _SELECTED_PIXEL
                        # intentionally
                        rgb_image_copy[i + ii, j + jj] = \
                            rgb_image_copy[wi + ii, wj + jj]
                        # Pixel covered don't revisit
                        component_mask_copy[i + ii, j + jj] = _UNSELECTED_PIXEL
    return rgb_image_copy


class RemoveInnerComponentOperation(ImageProcessingStep):
    """
    Operation identifies all internal
    """

    def __init__(
            self,
            extract_full_sign_pipeline: ProcessingStepSequence,
            extract_border_sign_pipeline: ProcessingStepSequence,
            valid_component_number: int,
            component_remove_idx: int,
            min_area_threshold: int,
            max_area_threshold: int,
            name: Optional[str] = None
    ):
        super().__init__(name)
        self._extract_full_sign_pipeline = extract_full_sign_pipeline
        self._extract_border_sign_pipeline = extract_border_sign_pipeline
        self._valid_component_number = valid_component_number
        self._fill_background_algorithm = FillUnselectedValues(
            origin_points=[point_2d.Point2D(0, 0)],  # Start filling from top left corner
            fill_value=BinaryMask.SELECTED_PIXEL_VALUE,
            empty_value=BinaryMask.UNSELECTED_PIXEL_VALUE,
        )
        self._component_remove_idx = component_remove_idx
        self._min_area_threshold = min_area_threshold
        self._max_area_threshold = max_area_threshold

    def apply(self, fn_input: RGBImage) -> Optional[RGBImage]:
        full_sign_mask: _BinaryMaskOrNone = \
            self._extract_full_sign_pipeline.apply(fn_input)
        border_sign_mask: _BinaryMaskOrNone = \
            self._extract_border_sign_pipeline.apply(fn_input)

        if full_sign_mask is None or border_sign_mask is None:
            return None

        inside_mask = full_sign_mask.xor(border_sign_mask)
        inside_sign: RGBImage = fn_input.get_crop(inside_mask)
        inside_sign_gs = inside_sign.to_grayscale()

        thresh: np.ndarray = cv2.adaptiveThreshold(
            inside_sign_gs.value, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10
        )  # Binarize the image in order to easily select the digits inside sign
        analysis = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        (totalLabels, label_ids, stats, centroid) = analysis
        areas = np.vstack(stats[:, cv2.CC_STAT_AREA][1:])  # Areas of all componentes

        # Subtract 1 to discard the background/ non-selected pixels
        # Component 0 is always the background
        if totalLabels - 1 != self._valid_component_number:
            return None

        # Note: I do this convoluted sorting to guarantee order as I am
        # unsure about the stability of the labels returned by cv2
        # The code below sorts the centroids and finds the label for the i-th
        # leftmost centroid as passed by user in constructor
        centroid_labels: np.ndarray = np.vstack([_ for _ in range(1, totalLabels)])
        labelled_centroids: np.ndarray = np.hstack([centroid[1:], centroid_labels, areas])
        # Sort centroids by X then Y
        labelled_centroids = labelled_centroids[np.lexsort([
            labelled_centroids[:, 1], labelled_centroids[:, 0]
        ])]
        seeked_label = labelled_centroids[self._component_remove_idx, 2]
        area_seeked_component = labelled_centroids[self._component_remove_idx, 3]
        if area_seeked_component < self._min_area_threshold or \
                area_seeked_component > self._max_area_threshold:
            # Image too noisy to threshold properly, we have selected
            # too many or too fex pixels due to heavy blur
            return None

        # Zero out all other pixels except of those belonging to component to remove
        component_to_remove_mask_np = np.uint8(thresh.copy())
        component_to_remove_mask_np[np.uint8(label_ids) != seeked_label] = _UNSELECTED_PIXEL

        # Check that the border is closed
        fill_in_background = self._fill_background_algorithm.apply(border_sign_mask)
        if np.all([fill_in_background == BinaryMask.SELECTED_PIXEL_VALUE]):
            # Border was not properly selected
            return None

        # Note: 'white' denotes here the pixels that are inside the road sign
        # but were assigned background by the threshold operations. We are going
        # to use their color in order to fill the area occupied by the
        # component we intend to remove. In order to do this, we select all other
        # pixels (background + components of interest) and negate them.
        component_pixels = np.float32(label_ids.copy())
        background_mask = full_sign_mask.negate()
        component_pixels[component_pixels != _UNSELECTED_PIXEL] = _SELECTED_PIXEL
        # Below are all pixels inside the sign that are not digits
        white_pixels_mask: np.ndarray = cv2.bitwise_not(
            cv2.bitwise_or(np.uint8(background_mask.value), np.uint8(component_pixels))
        )

        output = _cover_component_pixels(
            component_to_remove_mask_np, white_pixels_mask, fn_input.value
        )
        return RGBImage(output)
