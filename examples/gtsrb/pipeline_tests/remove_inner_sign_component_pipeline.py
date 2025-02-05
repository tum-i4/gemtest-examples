import numpy as np
import pytest

from examples.gtsrb.utils.deprecated_speed_sign.operation import remove_inner_sign_component_operation
from examples.gtsrb.utils.deprecated_speed_sign.visual_tools.processing_step.find_sign_by_border_color_sequence import \
    find_sign_by_border_color
from examples.gtsrb.utils.deprecated_speed_sign.visual_tools.processing_step.validate_full_sign_mask import \
    ValidateFullSignMaskOptions
from examples.gtsrb.utils.domain.color.hsv_color import COLOR_RANGES
from examples.gtsrb.utils.domain.dataset import gtsrb_load_train_default, gtsrb_load_test_default, \
    gtsrb_dataset
from examples.gtsrb.utils.domain.image.rgb_image import RGBImage

_ASSERT_SUCCESS_THRESHOLD = 20


@pytest.mark.skip(
    reason="Test broken after merge to main. GTSRB_Images train set should not be "
           "included in the repo.")
def test_remove_1_speed_sign_120():
    _default_train_strategy = gtsrb_load_train_default.GTSRBLoadTrainDefault(
        seeked_classes=[8],
        gtsrb_training_glob="../../assets/GTSRB_Images/Train/*/*",
    )
    _default_test_strategy = gtsrb_load_test_default.GTSRBLoadTestDefault(
        seeked_classes=[8],
        gtsrb_testing_glob="../../assets/GTSRB_Images/Test/*",
        gtsrb_testing_ground_truth_path="../../assets/GTSRB_Images/Test.csv",
    )
    ds = gtsrb_dataset.GTSRBDataset(
        train_load_strategy=_default_train_strategy,
        test_load_strategy=_default_test_strategy
    )
    border_mask_pipeline, background_mask_pipeline, full_sign_mask_pipeline = \
        find_sign_by_border_color(
            COLOR_RANGES.RED_COLOR.value,
            ValidateFullSignMaskOptions(
                min_area_selected_pipeline=0.1,
                max_area_selected_pipeline=0.8,
                tolerated_displacement_from_center=20,
                search_range_gaps_inside_center=20,
            )
        )
    remove_inner_op = remove_inner_sign_component_operation.RemoveInnerComponentOperation(
        extract_full_sign_pipeline=full_sign_mask_pipeline,
        extract_border_sign_pipeline=border_mask_pipeline,
        valid_component_number=4,
        component_remove_idx=0,
        min_area_threshold=500,
        max_area_threshold=650
    )
    count_valid = 0
    for img, _ in ds.train_iterator():
        img_np = np.array(img)
        output = remove_inner_op.apply(RGBImage(img_np))
        if output is not None:
            count_valid += 1
            if count_valid == _ASSERT_SUCCESS_THRESHOLD:
                break
    assert _ASSERT_SUCCESS_THRESHOLD == count_valid
