from typing import List

import numpy as np

import gemtest as gmt
from examples.gtsrb.metamorphic_tests.setup_image_classifier import export_data, \
    traffic_sign_visualizer, classifier_under_test
from examples.gtsrb.metamorphic_tests.setup_image_classifier import test_image_paths
from examples.gtsrb.utils.circle_sign.circle_red_sign import run_red_circle_processing, \
    run_red_circle_processing_realistic_font
from examples.gtsrb.utils.gtsrb_label_enum import GTSRBLabelEnum as GTSRB

number_of_test_cases = 10

create_non_existent_speed_sign = gmt.create_metamorphic_relation(
    name="create_non_existent_speed_sign",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases,
    number_of_sources=1,
    parameters={"sign": [5, 40, 90, 110, 130]}
)

# create_existing_speed_sign = gmt.create_metamorphic_relation(
#     name="create_existing_speed_sign",
#     data=test_image_paths,
#     testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
#     number_of_test_cases=number_of_test_cases,
#     number_of_sources=1,
#     parameters={"sign": [20, 30, 50, 60, 70, 80, 100, 120]}
# )

FONT_PATHS = [
    # Refer to https://en.wikipedia.org/wiki/DIN_1451
    # 1936 road sign font, hardly in use
    "./examples/gtsrb/assets/DIN1451-36breit.ttf",
    # Condensed version, in use
    "./examples/gtsrb/assets/DINEngschriftStd.otf",
    # Normal version, in use
    "./examples/gtsrb/assets/DINMittelschriftStd.otf",
]


@gmt.general_transformation(create_non_existent_speed_sign)
@gmt.randomized("font_number", gmt.RandInt(0, 2))
def overwrite_circle(mtc: gmt.MetamorphicTestCase, font_number: int) -> np.ndarray:
    img = mtc.source_input
    img_length = min(img[:, :, 0].shape)
    if img_length < 45:
        gmt.skip("Image too small")

    sign = mtc.parameters['sign']
    if 3 == font_number:
        final_image = run_red_circle_processing(img, str(sign))
    else:
        final_image = run_red_circle_processing_realistic_font(img, str(sign),
                                                               font_path=FONT_PATHS[font_number % len(FONT_PATHS)])

    if final_image is None:
        gmt.skip("Sign creation failed")
    return final_image


speed_signs = [
    GTSRB.MAX_SPEED_20,
    GTSRB.MAX_SPEED_30,
    GTSRB.MAX_SPEED_50,
    GTSRB.MAX_SPEED_60,
    GTSRB.MAX_SPEED_70,
    GTSRB.MAX_SPEED_80,
    GTSRB.MAX_SPEED_100,
    GTSRB.MAX_SPEED_120,
]


# TODO: do the same as above, return tuple of
# @gmt.general_transformation(create_existing_speed_sign)
# @gmt.randomized("text_index", gmt.RandInt(0, len(speed_signs) - 1))
# @gmt.randomized("font_number", gmt.RandInt(0, 2))
# def overwrite_circle(mtc: gmt.MetamorphicTestCase, font_number: int) -> np.ndarray:
#     img = mtc.source_input
#     img_length = min(img[:, :, 0].shape)
#     if img_length < 45:
#         gmt.skip("Image too small")
#
#     sign = mtc.parameters['sign']
#     if 3 == font_number:
#         final_image = run_red_circle_processing(img, str(sign))
#     else:
#        final_image = run_red_circle_processing_realistic_font(img, str(sign),
#                                                               font_path=FONT_PATHS[font_number % len(FONT_PATHS)])
#
#     if final_image is None:
#         gmt.skip("Sign creation failed")
#     return final_image


red_circle_keys = [
    GTSRB.MAX_SPEED_20,
    GTSRB.MAX_SPEED_30,
    GTSRB.MAX_SPEED_50,
    GTSRB.MAX_SPEED_60,
    GTSRB.MAX_SPEED_70,
    GTSRB.MAX_SPEED_80,
    GTSRB.MAX_SPEED_100,
    GTSRB.MAX_SPEED_120,
    GTSRB.NO_VEHICLES_PERMITTED_CIRCLE,
    GTSRB.NO_OVERTAKING,
    GTSRB.NO_OVERTAKING_FOR_TRUCKS
]


@gmt.valid_input(create_non_existent_speed_sign)
def fuse_triangles_input(y: int) -> bool:
    return y in red_circle_keys


# @gmt.valid_input(create_existing_speed_sign)
# def fuse_triangles_input(y: int) -> bool:
#     return y in red_circle_keys


@gmt.general_relation(create_non_existent_speed_sign)
def is_any_speed_sign(mtc: gmt.MetamorphicTestCase) -> bool:
    return mtc.followup_outputs[0] in speed_signs


# @gmt.general_relation(create_existing_speed_sign)
# def is_any_speed_sign(mtc: gmt.MetamorphicTestCase) -> bool:
#     return mtc.followup_output == mtc.followup_input[1]


@gmt.system_under_test(
    visualize_input=traffic_sign_visualizer.visualize_input,
    visualize_output=traffic_sign_visualizer.visualize_output,
    data_loader=gmt.load_image_resource,
    data_exporter=export_data
)
def test_image_classifier(image: np.ndarray) -> int:
    """Predict the traffic sign in an image"""
    return classifier_under_test.evaluate_image(image)


@gmt.systems_under_test_dynamic(
    data_loader=gmt.load_image_resource,
)
def test_mutant_image_classifier(images: List[np.ndarray], dynamic_sut) -> List[int]:
    with dynamic_sut:
        return dynamic_sut.execute(images)
