import os.path
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

import gemtest as gmt
from examples.gtsrb.metamorphic_tests.setup_image_classifier import test_image_paths
from examples.gtsrb.metamorphic_tests.setup_image_classifier import traffic_sign_visualizer, classifier_under_test
from examples.gtsrb.utils.circle_sign.circle_blue_sign import run_blue_circle_processing
from examples.gtsrb.utils.gtsrb_label_enum import GTSRBLabelEnum as GTSRB
from examples.gtsrb.metamorphic_tests.config import one_folders_up

asset_dir = Path(os.path.join(one_folders_up, "assets", "blue_circle"))

BLUE_SIGNS = {
    GTSRB.TURN_RIGHT: asset_dir / 'turn_right.png',
    GTSRB.TURN_LEFT: asset_dir / 'turn_left.png',
    GTSRB.GO_STRAIGHT: asset_dir / 'go_straight.png',
    GTSRB.GO_STRAIGHT_OR_RIGHT: asset_dir / 'go_straight_or_right.png',
    GTSRB.GO_STRAIGHT_OR_LEFT: asset_dir / 'go_straight_or_left.png',
    GTSRB.DRIVE_RIGHT: asset_dir / 'drive_right.png',
    GTSRB.DRIVE_LEFT: asset_dir / 'drive_left.png',
    GTSRB.ROUNDABOUT: asset_dir / 'roundabout.png',
}
BLUE_SIGN_KEYS = list(BLUE_SIGNS.keys())

create_circle_blue = gmt.create_metamorphic_relation(
    name="create_circle_blue",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    parameters={"target_label": BLUE_SIGN_KEYS},
)


@lru_cache
def load_cached(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


@gmt.general_transformation(create_circle_blue)
def overwrite_circle(mtc: gmt.MetamorphicTestCase) -> np.ndarray:
    img = mtc.source_inputs[0]
    img_length = min(img[:, :, 0].shape)

    if img_length < 45:
        gmt.skip("Image too small")

    target_label = mtc.parameters['target_label']
    overlay_img = load_cached(BLUE_SIGNS[target_label])
    final_image = run_blue_circle_processing(img.copy(), overlay_img, confidence_thr=0.6)

    if final_image is None:
        gmt.skip("Sign creation failed")

    return final_image


@gmt.valid_input(create_circle_blue)
def is_blue_sign_label(y: int) -> bool:
    return y in BLUE_SIGNS


@gmt.general_relation(create_circle_blue)
def is_specified_blue_sign(mtc: gmt.MetamorphicTestCase) -> bool:
    return mtc.followup_output == mtc.parameters['target_label']


@gmt.system_under_test(
    visualize_input=traffic_sign_visualizer.visualize_input,
    visualize_output=traffic_sign_visualizer.visualize_output,
    data_loader=gmt.load_image_resource,
)
def test_image_classifier(image: np.ndarray) -> int:
    """Predict the traffic sign in an image"""
    return classifier_under_test.evaluate_image(image)

# @gmt.systems_under_test_dynamic(
#     data_loader=gmt.load_image_resource,
# )
# def test_mutant_image_classifier(images: List[np.ndarray], dynamic_sut) -> List[int]:
#     with dynamic_sut:
#         return dynamic_sut.execute(images)
