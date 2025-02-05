from typing import Dict, List

import numpy as np
import tensorflow as tf

import examples.svhn.data.utils as image_loader
import gemtest as gmt

tf2 = tf.compat.v2
from examples.svhn.metamorphic_tests.setup_svhn_classifier import test_image_paths, svhn_visualizer, \
    classifier_under_test
import imgaug.augmenters as iaa

number_of_test_cases = 10  # it only used with TestingStrategy.SIMPLE
img_channels = 3

image_180_rotate = gmt.create_metamorphic_relation(
    name="image_180_rotate_preserving",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_horizontal_flip = gmt.create_metamorphic_relation(
    name="image_horizontal_flip",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)


@gmt.transformation(image_180_rotate)
def image_180_rotate_transform(image: np.ndarray) -> np.ndarray:
    seq = iaa.Sequential([iaa.Rot90(2)])
    aug_image = seq(image=image)
    return aug_image


@gmt.transformation(image_horizontal_flip)
def image_horizontal_flip_transform(image: np.ndarray) -> np.ndarray:
    seq = iaa.Sequential([iaa.Flipud()])
    aug_image = seq(image=image)
    return aug_image


@gmt.relation(image_180_rotate)
def image_180_rotate_relation(source_output: int, followup_output: int):
    mapping: Dict[int, int] = {6: 9, 9: 6}
    x_hat: int = mapping.get(source_output, source_output)
    return gmt.equality(x_hat, followup_output)


@gmt.relation(image_horizontal_flip)
def image_horizontal_flip_transform(source_output: int, followup_output: int):
    # flip only 2 -> [5,6]
    mapping: Dict[int, List[int]] = {2: [5]}
    x_hat: List[int] = mapping.get(source_output, source_output)
    return gmt.equality(x_hat[0], followup_output)


@gmt.valid_input(image_180_rotate)
def image_180_rotate_input(y: int) -> bool:
    valids = [6, 9]
    return y in valids


@gmt.valid_input(image_horizontal_flip)
def image_horizontal_flip_input(y: int) -> bool:
    valids = [2, ]
    return y in valids


@gmt.system_under_test(
    visualize_input=svhn_visualizer.visualize_input,
    visualize_output=svhn_visualizer.visualize_output,
    data_loader=image_loader.load_image_resource,
    # data_exporter=export_data
)
def test_image_classifier(image: np.ndarray) -> int:
    """Predict the traffic sign in an image"""
    return classifier_under_test.evaluate_image(image)
