import typing

import numpy as np
import pytest

from examples.gtsrb.pipeline_tests import change_background_pipeline
from examples.gtsrb.utils.deprecated_speed_sign.visual_tools.processing_step.find_sign_by_border_color_sequence import \
    find_sign_by_border_color
from examples.gtsrb.utils.deprecated_speed_sign.visual_tools.processing_step.validate_full_sign_mask import \
    ValidateFullSignMaskOptions
from examples.gtsrb.utils.domain import types
from examples.gtsrb.utils.domain.dataset import gtsrb_load_test_default, \
    gtsrb_dataset, gtsrb_load_train_default
from examples.gtsrb.utils.domain.image import rgb_image

# How many images should be correctly tagged by the pipeline
# to correctly mark the class as correct
_SUCCESS_THRESHOLD = 5


def _evaluate(image_generator, mask_extraction_pipeline):
    total_count, valid_count = 0, None
    for img, _ in image_generator():
        total_count += 1
        mask = mask_extraction_pipeline.apply(img)
        total_count += 1
        if mask is not None:
            if valid_count is None:
                valid_count = 0
            valid_count += 1
        if valid_count == _SUCCESS_THRESHOLD:
            break
    return total_count, valid_count


@pytest.mark.skip(
    reason="Test broken after merge to main. GTSRB_Images train set should not be "
           "included in the repo.")
@pytest.mark.parametrize("test_values", change_background_pipeline.configure_test_values())
def test_select_border_works(
        test_values: typing.List[change_background_pipeline.CLASS_HSV_COLOR_TUPLE]
):
    def generate_rgb_image(image_gen: types.DataIterator):
        """Wrapper that converts all Numpy arrays to RGB images."""

        def inner_generator() -> typing.Generator[rgb_image.RGBImage, None, None]:
            for img, _ in image_gen:
                img_np = np.array(img)
                yield rgb_image.RGBImage(img_np)

        return inner_generator

    img_class, border_hsv_color = test_values
    _default_train_strategy = gtsrb_load_train_default.GTSRBLoadTrainDefault(
        seeked_classes=[img_class],
        gtsrb_training_glob="../../../../assets/GTSRB_Images/Train/*/*",
    )
    _default_test_strategy = gtsrb_load_test_default.GTSRBLoadTestDefault(
        seeked_classes=[img_class],
        gtsrb_testing_glob="../../../../assets/GTSRB_Images/Test/*",
        gtsrb_testing_ground_truth_path="../../../../assets/GTSRB_Images/Test.csv",
    )
    ds = gtsrb_dataset.GTSRBDataset(
        train_load_strategy=_default_train_strategy,
        test_load_strategy=_default_test_strategy
    )
    _, _, full_sign_mask = find_sign_by_border_color(
        border_hsv_color,
        ValidateFullSignMaskOptions(
            min_area_selected_pipeline=0.2,
            max_area_selected_pipeline=0.6,
            tolerated_displacement_from_center=20,
            search_range_gaps_inside_center=25,
        )
    )
    total_count, valid_count = _evaluate(
        generate_rgb_image(ds.train_iterator()), full_sign_mask
    )
    assert valid_count == _SUCCESS_THRESHOLD, \
        f'Failed for {img_class} {border_hsv_color}'
