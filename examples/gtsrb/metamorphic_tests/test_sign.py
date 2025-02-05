import os
from typing import List

import numpy as np

import gemtest as gmt
from examples.gtsrb.metamorphic_tests.setup_image_classifier import test_image_paths, \
    traffic_sign_visualizer, export_data, classifier_under_test
from examples.gtsrb.utils.deprecated_speed_sign.operation.write_speed_operation import \
    WriteSpeedOperation
from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.gtsrb_label_enum import GTSRBLabelEnum as GTSRB

number_of_test_cases = 10
module_folder = os.path.dirname(__file__)
one_folders_up = os.path.dirname(module_folder)

'''Creates new images of speed signs. 
The speed limit text is created by using the fonts used in the real world for the traffic signs. 
The speed limit is written on images of the class "No vehicles permitted (circle)"'''
# create signs that do not exist in the dataset
create_sign = gmt.create_metamorphic_relation(
    name="create_sign",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases,
    parameters={"sign": [90, 110, 130]})

# create sign in the dataset
# create_20_sign = metamorphic("create_20_sign")
# create_100_sign = metamorphic("create_100_sign")

FONT_PATHS = [
    # Refer to https://en.wikipedia.org/wiki/DIN_1451
    # 1936 road sign font, hardly in use
    os.path.join(one_folders_up, "assets/DIN1451-36breit.ttf"),
    # Condensed version, in use
    os.path.join(one_folders_up, "assets/DINEngschriftStd.otf"),
    # Normal version, in use
    os.path.join(one_folders_up, "assets/DINMittelschriftStd.otf"),
]


@gmt.general_transformation(create_sign)
@gmt.randomized("font_number", gmt.RandInt(0, 2))
def create_signs(mtc: gmt.MetamorphicTestCase, font_number: int) -> np.ndarray:
    image = mtc.source_input
    sign = mtc.parameters['sign']
    sign_generator = \
        WriteSpeedOperation(str(sign),
                            control_image_path=os.path.join(one_folders_up, "assets/forbidden.png"),
                            font_path=FONT_PATHS[font_number % len(FONT_PATHS)])
    img = RGBImage(image)
    result = sign_generator.apply(img)
    if result is None:
        gmt.skip("Sign creation failed")
    return result.value


sing_creation_valid_outputs = [
    GTSRB.MAX_SPEED_20,
    GTSRB.MAX_SPEED_30,
    GTSRB.MAX_SPEED_50,
    GTSRB.MAX_SPEED_60,
    GTSRB.MAX_SPEED_70,
    GTSRB.MAX_SPEED_80,
    GTSRB.NO_MAX_SPEED_80,
    GTSRB.MAX_SPEED_100,
    GTSRB.MAX_SPEED_120
]


@gmt.general_relation(create_sign)
def is_any_speed_sign(mtc: gmt.MetamorphicTestCase) -> bool:
    return mtc.followup_outputs[0] in sing_creation_valid_outputs


@gmt.valid_input(create_sign)
def input_forbidden_sign(source_output: int) -> bool:
    # only allow the "No vehicles permitted (circle)" sign
    return source_output == GTSRB.NO_VEHICLES_PERMITTED_CIRCLE


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
