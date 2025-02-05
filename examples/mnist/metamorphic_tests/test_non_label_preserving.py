from typing import Dict

import numpy as np
import tensorflow as tf

import examples.mnist.data.data_loaders as image_loader
import examples.mnist.utils.augmentationUtils as utils  # https://imgaug.readthedocs.io/en/latest/
import gemtest as gmt
from examples.mnist.metamorphic_tests.setup_torch_classifier import (test_image_paths, mnist_visualizer,
                                                                     classifier_under_test, export_data)

tf2 = tf.compat.v2

number_of_test_cases = 10
img_channels = 3

'''creates a 6 from a 9 and vice versa by a vertical flip'''
image_vertical_flip = gmt.create_metamorphic_relation(
    name="image_vertical_flip 6->9",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''creates a 5 from a 2 by a horizontal flip'''
image_horizontal_flip = gmt.create_metamorphic_relation(
    name="image_horizontal_flip 2->5",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''creates a 5 or 6 from a 2 by a horizontal flip'''
image_horizontal_flip2 = gmt.create_metamorphic_relation(
    name="image_horizontal_flip 2->5 or 6",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)


@gmt.transformation(image_vertical_flip)
def image_vertical_transform(image: np.ndarray) -> np.ndarray:
    aug_image = utils.rotate_by_factors_of_90(image, 2)
    return aug_image


@gmt.general_transformation(image_horizontal_flip)
def image_horizontal_flip_transform(mtc: gmt.MetamorphicTestCase):
    # check if digit-two is well-shaped (explained in 1-4)
    # 1. cut the lower 1/3 of the picture
    image = mtc.source_inputs[0]
    height_to_remove = int(image.shape[0] * 1 / 2)
    mask = np.ones_like(image)
    mask[:height_to_remove, :, :] = 0
    image_one = image * mask
    # 2. rotate it by 90 degree
    image_one = utils.rotate_by_factors_of_90(image_one, 1)
    # 3. flip it horizontally
    image_one = utils.horizontal_flip(image_one)
    # 4. check if the model predicts it to be one/seven with high-enough prob
    # todo evaluate_image call might be a MR with multiple followup inputs
    predicted_label = classifier_under_test.evaluate_image(image_one)
    expected_label = [1, 7]
    if predicted_label not in expected_label:
        gmt.skip("The second follow-up input seems not to be a 1 or 7")
    aug_image = utils.horizontal_flip(image)
    return image_one, aug_image


@gmt.transformation(image_horizontal_flip2)
def image_horizontal_flip_transform2(image: np.ndarray) -> np.ndarray:
    aug_image = utils.horizontal_flip(image)
    return aug_image


@gmt.relation(image_vertical_flip)
def image_vertical_flip_relation(source_output: int, followup_output: int):
    mapping: Dict[int, int] = {6: 9, 9: 6}
    x_hat: int = mapping.get(source_output, source_output)
    return gmt.equality(x_hat, followup_output)


@gmt.general_relation(image_horizontal_flip)
def image_horizontal_flip_relation(mtc: gmt.MetamorphicTestCase):
    source_output = mtc.source_outputs[0]
    followup_output = mtc.followup_outputs[1]
    mapping: Dict[int, int] = {2: 5}
    x_hat: int = mapping.get(source_output, source_output)
    return gmt.equality(x_hat, followup_output)


@gmt.relation(image_horizontal_flip2)
def image_horizontal_flip_relation2(source_output: int, followup_output: int):
    return gmt.equality(followup_output, 5) or gmt.equality(followup_output, 6)


@gmt.valid_input(image_vertical_flip)
def image_vertical_flip_input(y: int) -> bool:
    valids = [6, 9]
    return y in valids


@gmt.valid_input(image_horizontal_flip)
def image_horizontal_flip_input2(y: int) -> bool:
    valids = [2]
    return y in valids


@gmt.valid_input(image_horizontal_flip2)
def image_horizontal_flip_input2(y: int) -> bool:
    valids = [2]
    return y in valids


@gmt.system_under_test(
    visualize_input=mnist_visualizer.visualize_input,
    visualize_output=mnist_visualizer.visualize_output,
    data_loader=image_loader.load_image_resource,
    data_exporter=export_data
)
def test_image_classifier(image: np.ndarray) -> int:
    """Predict the traffic sign in an image"""
    return classifier_under_test.evaluate_image(image)
