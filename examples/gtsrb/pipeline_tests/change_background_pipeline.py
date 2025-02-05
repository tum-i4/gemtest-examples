import itertools
import typing

import numpy as np
import pytest

from examples.gtsrb.utils.deprecated_speed_sign.operation import change_background_operation
from examples.gtsrb.utils.deprecated_speed_sign.visual_tools.processing_step.validate_full_sign_mask import \
    ValidateFullSignMaskOptions
from examples.gtsrb.utils.domain.color.hsv_color import HSVColor, COLOR_RANGES
from examples.gtsrb.utils.domain.dataset import gtsrb_load_train_default, gtsrb_dataset, \
    gtsrb_load_test_default
from examples.gtsrb.utils.domain.dataset.gtsrb_dataset import GTSRBDataset
from examples.gtsrb.utils.domain.image.rgb_image import RGBImage

_SUCCESS_THRESHOLD = 5

CLASS_HSV_COLOR_TUPLE = typing.Tuple[int, typing.Tuple[HSVColor]]


def configure_test_values() -> typing.List[CLASS_HSV_COLOR_TUPLE]:
    """Return class sign and the HSV color of the border to select it."""
    values = []
    for image_cls, border_hsv_color in itertools.product(
            GTSRBDataset.RED_BORDER_CLASSES, [COLOR_RANGES.RED_COLOR.value]
    ):
        values.append((image_cls, border_hsv_color))
    for image_cls, border_hsv_color in itertools.product(
            GTSRBDataset.BLUE_BORDER_CLASSES, [COLOR_RANGES.BLUE_COLOR.value]
    ):
        values.append((image_cls, border_hsv_color))
    return values


@pytest.mark.skip(
    reason="Test broken after merge to main. GTSRB_Images train set should not be "
           "included in the repo.")
@pytest.mark.parametrize("test_values", configure_test_values())
def test_change_background_operation(
        test_values: typing.List[CLASS_HSV_COLOR_TUPLE]
):
    img_class, border_hsv_color = test_values
    _default_train_strategy = gtsrb_load_train_default.GTSRBLoadTrainDefault(
        seeked_classes=[img_class],
        gtsrb_training_glob="../../assets/GTSRB_Images/Train/*/*",
    )
    _default_test_strategy = gtsrb_load_test_default.GTSRBLoadTestDefault(
        seeked_classes=[img_class],
        gtsrb_testing_glob="../../assets/GTSRB_Images/Test/*",
        gtsrb_testing_ground_truth_path="../../assets/GTSRB_Images/Test.csv",
    )
    gtsrb_ds = gtsrb_dataset.GTSRBDataset(
        train_load_strategy=_default_train_strategy,
        test_load_strategy=_default_test_strategy
    )
    valid_counter = 0
    for img_val, _ in gtsrb_ds.train_iterator():
        # Signs in these classes are un-centered from the true middle
        # of the image, tolerate larger skews
        img_np = np.array(img_val)
        result = change_background_operation.ChangeBackgroundOperation(
            border_hsv_color, color_intensity=0.6,
            full_sign_validation_options=ValidateFullSignMaskOptions(
                min_area_selected_pipeline=0.2,
                max_area_selected_pipeline=0.8,
                tolerated_displacement_from_center=30,
                search_range_gaps_inside_center=20,
            )
        ).apply(RGBImage(img_np))
        if result is not None:
            valid_counter += 1
        if valid_counter == _SUCCESS_THRESHOLD:
            break
    assert valid_counter == _SUCCESS_THRESHOLD
