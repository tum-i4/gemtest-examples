import cv2
import numpy as np
import tensorflow as tf

import examples.mnist.data.data_loaders as image_loader
import examples.mnist.utils.augmentationUtils as utils  # https://imgaug.readthedocs.io/en/latest/
import gemtest as gmt
from examples.mnist.metamorphic_tests.setup_torch_classifier import (test_image_paths, mnist_visualizer,
                                                                     classifier_under_test, export_data)

tf2 = tf.compat.v2

number_of_test_cases = 10

'''creates a 9 from a 0&1 by resizing and overlaying both images'''
image_overlay10 = gmt.create_metamorphic_relation(
    name="image_overlay10 1&0->9",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases,
    number_of_sources=2)

'''creates a 2 from a 5 by a horizontal flip'''
image_horizontal_flip5 = gmt.create_metamorphic_relation(
    name="image_horizontal_flip 5->2",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)


@gmt.general_transformation(image_overlay10)
def image_overlay10_transform(mtc: gmt.MetamorphicTestCase):
    if mtc.source_outputs[0] == 0 and mtc.source_outputs[1] == 1:
        input_0 = mtc.source_inputs[0]
        input_1 = mtc.source_inputs[1]

    elif mtc.source_outputs[1] == 0 and mtc.source_outputs[0] == 1:
        input_0 = mtc.source_inputs[1]
        input_1 = mtc.source_inputs[0]
    else:
        return gmt.skip("Source inputs are not of label \"0\" and \"1\"")

    # horizontal flip if the one has negative incline
    if not detect_positive_incline(input_1):
        input_1 = utils.horizontal_flip(input_1)
    # squish the digits
    input_0 = utils.scale(input_0, [0.5, 0.5], [0.5, 0.5])
    input_1 = utils.scale(input_1, [0.5, 0.5], [0.5, 0.5])
    # shift the zero to the top
    _, _, top_0, _ = utils.get_margins(input_0)
    aug_image = utils.shift(input_0, (0, 0), (-top_0, -top_0))
    composite_image = np.maximum(aug_image, input_1)
    # shift the composite image towards center
    composite_image = utils.shift(composite_image, (0, 0), (int(top_0 / 2), int(top_0 / 2)))
    return composite_image


@gmt.transformation(image_horizontal_flip5)
def image_horizontal_flip_transform(image: np.ndarray) -> np.ndarray:
    aug_image = utils.horizontal_flip(image)
    return aug_image


@gmt.general_relation(image_overlay10)
def image_overlay10_relation(mtc: gmt.MetamorphicTestCase) -> bool:
    source_output1 = mtc.source_outputs[0]
    source_output2 = mtc.source_outputs[1]
    if source_output1 == source_output2:
        # return true if both source-outputs 1 or both 0
        return True
    return gmt.equality(mtc.followup_outputs[0], 9)


@gmt.relation(image_horizontal_flip5)
def image_horizontal_flip_relation(source_output: int, followup_output: int):
    return followup_output == 2


@gmt.valid_input(image_overlay10)
def image_overlay10_input(source_output: int) -> bool:
    return source_output == 0 or source_output == 1


@gmt.valid_input(image_horizontal_flip5)
def image_horizontal_flip_input(y: int) -> bool:
    return y == 5


def detect_positive_incline(image):
    # Threshold the image
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours
    for contour in contours:
        # Fit a line to the contour
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        # Check the slope (positive incline)
        if vy > 0:
            return False

    return True


@gmt.system_under_test(
    visualize_input=mnist_visualizer.visualize_input,
    visualize_output=mnist_visualizer.visualize_output,
    data_loader=image_loader.load_image_resource,
    data_exporter=export_data
)
def test_image_classifier(image: np.ndarray) -> int:
    """Predict the traffic sign in an image"""
    return classifier_under_test.evaluate_image(image)
