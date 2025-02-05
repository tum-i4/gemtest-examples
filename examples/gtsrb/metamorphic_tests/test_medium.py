import math
from typing import Dict, List

import albumentations
import numpy as np
from PIL import ImageEnhance, Image

import gemtest as gmt
from examples.gtsrb.metamorphic_tests.setup_image_classifier import test_image_paths, \
    traffic_sign_visualizer, export_data, classifier_under_test
from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.gtsrb_label_enum import GTSRBLabelEnum as GTSRB
from examples.gtsrb.utils.processing_steps.rotate_and_rescale import RotateAndRescale

number_of_test_cases = 10
'''Contains relations which change the label of an image.'''

black_and_white_valid_inputs = [
    GTSRB.MAX_SPEED_80,
    GTSRB.NO_MAX_SPEED_80,
    GTSRB.NO_OVERTAKING,
    GTSRB.NO_OVERTAKING_FOR_TRUCKS,
    GTSRB.NO_VEHICLES_PERMITTED_CIRCLE,
    GTSRB.NO_MAX_SPEED,
    GTSRB.PASSING_ALLOWED,
    GTSRB.PASSING_ALLOWED_FOR_TRUCKS
]
h_flip_simple_valid_inputs = [
    GTSRB.PRIORITY_IN_TRAFFIC_NEXT_CROSSING,
    GTSRB.PRIORITY_IN_TRAFFIC_ROAD,
    GTSRB.YIELD,
    GTSRB.STOP,
    GTSRB.NO_VEHICLES_PERMITTED_CIRCLE,
    GTSRB.TRUCK_DRIVING_LEFT,
    GTSRB.NO_VEHICLES_PERMITTED_WHITE_BOX,
    GTSRB.WARNING,
    GTSRB.DANGEROUS_DOUBLE_CURVE,
    GTSRB.BUMPY_ROAD,
    GTSRB.SLIPPERY_ROAD,
    GTSRB.ROAD_NARROWS_FROM_RIGHT,
    GTSRB.CONSTRUCTION_SITE,
    GTSRB.TRAFFIC_LIGHT_AHEAD,
    GTSRB.PEDESTRIAN,
    GTSRB.CHILDREN_CROSSING,
    GTSRB.BICYCLE_AHEAD,
    GTSRB.SNOWY_ROAD,
    GTSRB.DEER_CROSSING,
    GTSRB.GO_STRAIGHT,
    GTSRB.ROUNDABOUT
]

h_flip_advanced_relation_mapping = {
    GTSRB.DANGEROUS_CURVE_TO_LEFT: GTSRB.DANGEROUS_CURVE_TO_RIGHT,
    GTSRB.DANGEROUS_CURVE_TO_RIGHT: GTSRB.DANGEROUS_CURVE_TO_LEFT,
    GTSRB.TURN_RIGHT: GTSRB.TURN_LEFT,
    GTSRB.TURN_LEFT: GTSRB.TURN_RIGHT,
    GTSRB.GO_STRAIGHT_OR_RIGHT: GTSRB.GO_STRAIGHT_OR_LEFT,
    GTSRB.GO_STRAIGHT_OR_LEFT: GTSRB.GO_STRAIGHT_OR_RIGHT,
    GTSRB.DRIVE_RIGHT: GTSRB.DRIVE_LEFT,
    GTSRB.DRIVE_LEFT: GTSRB.DRIVE_RIGHT
}

vertical_flip_valid_inputs = [GTSRB.PRIORITY_IN_TRAFFIC_ROAD, GTSRB.STOP, GTSRB.NO_VEHICLES_PERMITTED_CIRCLE]

rot_90_clockwise_relation_mapping = {
    GTSRB.PRIORITY_IN_TRAFFIC_ROAD: GTSRB.PRIORITY_IN_TRAFFIC_ROAD,
    GTSRB.STOP: GTSRB.STOP,
    GTSRB.NO_VEHICLES_PERMITTED_CIRCLE: GTSRB.NO_VEHICLES_PERMITTED_CIRCLE,
    GTSRB.DRIVE_RIGHT: GTSRB.DRIVE_LEFT
}

rot_90_counterclockwise_relation_mapping = {
    GTSRB.PRIORITY_IN_TRAFFIC_ROAD: GTSRB.PRIORITY_IN_TRAFFIC_ROAD,
    GTSRB.STOP: GTSRB.STOP,
    GTSRB.NO_VEHICLES_PERMITTED_CIRCLE: GTSRB.NO_VEHICLES_PERMITTED_CIRCLE,
    GTSRB.DRIVE_LEFT: GTSRB.DRIVE_RIGHT
}

rot_135_clockwise_relation_mapping = {
    GTSRB.STOP: GTSRB.STOP,
    GTSRB.NO_VEHICLES_PERMITTED_CIRCLE: GTSRB.NO_VEHICLES_PERMITTED_CIRCLE,
    GTSRB.GO_STRAIGHT: GTSRB.DRIVE_RIGHT,
    GTSRB.DRIVE_LEFT: GTSRB.GO_STRAIGHT
}

rot_135_counterclockwise_relation_mapping = {
    GTSRB.STOP: GTSRB.STOP,
    GTSRB.NO_VEHICLES_PERMITTED_CIRCLE: GTSRB.NO_VEHICLES_PERMITTED_CIRCLE,
    GTSRB.GO_STRAIGHT: GTSRB.DRIVE_LEFT,
    GTSRB.DRIVE_RIGHT: GTSRB.GO_STRAIGHT
}
'''Rotates the image between -30 to 30 degrees.'''
rotation = gmt.create_metamorphic_relation(
    name="rotation",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''Turns the image into a grayscale image.'''
black_and_white_specific = gmt.create_metamorphic_relation(
    name="black_and_white_specific",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''Bends the image along the x-axis. Creates an arc left'''
bend_image_vertically = gmt.create_metamorphic_relation(
    name="bend_image_vertically",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''Bends the image along the y-axis Creates an arc top.'''
bend_image_horizontally = gmt.create_metamorphic_relation(
    name="bend_image_horizontally",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''Bends the image along both axis.'''
bend_image_twice = gmt.create_metamorphic_relation(
    name="bend_image_twice",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''Mirror the image horizontally. No label changes allowed.'''
horizontal_flip_simple = gmt.create_metamorphic_relation(
    name="horizontal_flip_simple",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''Mirror the image horizontally. Only images with label changes allowed.'''
horizontal_flip_advanced = gmt.create_metamorphic_relation(
    name="horizontal_flip_advanced",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''Mirror the image vertically.'''
vertical_flip = gmt.create_metamorphic_relation(
    name="vertical_flip",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''Mirror the image horizontally and equalize the image histogram. No label changes allowed.'''
pair_simple = gmt.create_metamorphic_relation(
    name="hflip_equalize_simple",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

''' Mirror the image horizontally and equalize the image histogram. 
Only images with label change allowed.'''
pair_advanced = gmt.create_metamorphic_relation(
    name="hflip_equalize_advanced",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''Rotate the image by 135° clockwise.'''
rotation_135deg_clockwise = gmt.create_metamorphic_relation(
    name="rotation_135deg_clockwise",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''Rotate the image by 135° counterclockwise.'''
rotation_135deg_counterclockwise = gmt.create_metamorphic_relation(
    name="rotation_135deg_counterclockwise",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''Rotate the image by 90° clockwise.'''
rotation_90deg_clockwise = gmt.create_metamorphic_relation(
    name="rotation_90deg_clockwise",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''Rotate the image by 90° counterclockwise.'''
rotation_90deg_counterclockwise = gmt.create_metamorphic_relation(
    name="rotation_90deg_counterclockwise",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)


@gmt.transformation(rotation)
@gmt.randomized("angle_in_degrees", gmt.RandInt(-30, 30))
def rotate(image: np.ndarray, angle_in_degrees: int) -> np.ndarray:
    result = RotateAndRescale(angle_in_degrees=angle_in_degrees).apply(RGBImage(image))
    return result.value


def _bend_image_vertically(image: np.ndarray, bend: int) -> np.ndarray:
    rows = image.shape[0]
    cols = image.shape[1]
    img_output = np.zeros(image.shape, dtype=image.dtype)

    for i in range(rows):
        for j in range(cols):
            offset_x = int(bend * math.sin(2 * 3.14 * i / (2 * cols)) - bend / 2)
            if j + offset_x < cols:
                img_output[i, j] = image[i, (j + offset_x) % cols]
            else:
                img_output[i, j] = 0

    return img_output


def _bend_image_horizontally(image: np.ndarray, bend: int) -> np.ndarray:
    rows = image.shape[0]
    cols = image.shape[1]
    img_output = np.zeros(image.shape, dtype=image.dtype)

    for i in range(rows):
        for j in range(cols):
            offset_y = int(bend * math.sin(2 * 3.14 * j / (2 * rows)) - bend / 2)
            if i + offset_y < rows:
                img_output[i, j] = image[(i + offset_y) % rows, j]
            else:
                img_output[i, j] = 0

    return img_output


@gmt.transformation(black_and_white_specific)
def make_blackandwhite(image: np.ndarray) -> np.ndarray:
    img = Image.fromarray(image)
    enhancer = ImageEnhance.Color(img)
    return np.array(enhancer.enhance(0))


@gmt.valid_input(black_and_white_specific)
def input_black_and_white(y: int) -> bool:
    valids = black_and_white_valid_inputs
    return y in valids


@gmt.transformation(bend_image_vertically)
@gmt.randomized("bend", gmt.RandInt(10, 15))
def bend_image_vertically(image: np.ndarray, bend: int) -> np.ndarray:
    return _bend_image_vertically(image, bend)


@gmt.transformation(bend_image_horizontally)
@gmt.randomized("bend", gmt.RandInt(5, 10))
def bend_image_horizontally(image: np.ndarray, bend: int) -> np.ndarray:
    return _bend_image_horizontally(image, bend)


@gmt.transformation(bend_image_twice)
@gmt.randomized("vertical_bend", gmt.RandInt(10, 15))
@gmt.randomized("horizontal_bend", gmt.RandInt(5, 10))
def bend_image_twice(image: np.ndarray, vertical_bend: int,
                     horizontal_bend: int) -> np.ndarray:
    image_horizontal = _bend_image_horizontally(image, horizontal_bend)
    return _bend_image_vertically(image_horizontal, vertical_bend)


@gmt.transformation(horizontal_flip_simple, horizontal_flip_advanced)
def album_horizonflip(image: np.ndarray) -> np.ndarray:
    image_transform = albumentations.HorizontalFlip(p=1)
    return image_transform.apply(image)


@gmt.transformation(pair_simple, pair_advanced)
def album_horizonflip_equalize(image: np.ndarray) -> np.ndarray:
    image_transform = albumentations.HorizontalFlip(p=1)
    horizonflip = image_transform.apply(image)
    image_transform = albumentations.Equalize(p=1)
    return image_transform.apply(horizonflip)


@gmt.transformation(vertical_flip)
def album_verticalflip(image: np.ndarray) -> np.ndarray:
    image_transform = albumentations.VerticalFlip(p=1)
    return image_transform.apply(image)


@gmt.relation(horizontal_flip_simple, pair_simple)
def flip_sign_horizontal_simple(f_x: int, f_xt: int) -> bool:
    return gmt.equality(f_x, f_xt)


@gmt.valid_input(horizontal_flip_simple, pair_simple)
def input_horizontal_flip(y: int) -> bool:
    valids = h_flip_simple_valid_inputs
    return y in valids


@gmt.relation(horizontal_flip_advanced, pair_advanced)
def flip_sign_horizontal_advanced(f_x: int, f_xt: int) -> bool:
    mapping: Dict[int, int] = h_flip_advanced_relation_mapping
    x_hat: int = mapping.get(f_x, f_x)
    return gmt.equality(x_hat, f_xt)


@gmt.valid_input(horizontal_flip_advanced, pair_advanced)
def input_horizontal_flip_advanced(y: int) -> bool:
    valids = h_flip_advanced_relation_mapping.keys()
    return y in valids


@gmt.relation(vertical_flip)
def flip_sign_vertical(f_x: int, f_xt: int) -> bool:
    return gmt.equality(f_x, f_xt)


@gmt.valid_input(vertical_flip)
def input_vertical_flip(y: int) -> bool:
    valids = vertical_flip_valid_inputs

    return y in valids


@gmt.relation(rotation_135deg_counterclockwise)
def rotation_135deg_counterclockwise_rel(x: int, y: int) -> bool:
    mapping: Dict[int, int] = rot_135_counterclockwise_relation_mapping
    x_hat: int = mapping.get(x, x)
    return gmt.equality(x_hat, y)


@gmt.relation(rotation_135deg_clockwise)
def rotation_135deg_clockwise_rel(x: int, y: int) -> bool:
    mapping: Dict[int, int] = rot_135_clockwise_relation_mapping
    x_hat: int = mapping.get(x, x)
    return gmt.equality(x_hat, y)


@gmt.valid_input(rotation_135deg_counterclockwise)
def rotation_135deg_counterclockwise_input(y: int) -> bool:
    valids = rot_135_counterclockwise_relation_mapping.keys()
    return y in valids


@gmt.valid_input(rotation_135deg_clockwise)
def rotation_135deg_clockwise_input(y: int) -> bool:
    valids = rot_135_clockwise_relation_mapping.keys()
    return y in valids


@gmt.transformation(rotation_135deg_clockwise)
@gmt.randomized("angle_in_degrees", gmt.RandInt(125, 145))
def rotate_135deg_clockwise(image: np.ndarray, angle_in_degrees: int) -> np.ndarray:
    result = RotateAndRescale(angle_in_degrees=angle_in_degrees).apply(RGBImage(image))
    return result.value


@gmt.transformation(rotation_135deg_counterclockwise)
@gmt.randomized("angle_in_degrees", gmt.RandInt(-145, -125))
def rotate_135deg_counterclockwise(image: np.ndarray, angle_in_degrees: int) -> np.ndarray:
    result = RotateAndRescale(angle_in_degrees=angle_in_degrees).apply(RGBImage(image))
    return result.value


@gmt.transformation(rotation_90deg_clockwise)
@gmt.randomized("angle_in_degrees", gmt.RandInt(80, 100))
def rotate_90deg_clockwise(image: np.ndarray, angle_in_degrees: int) -> np.ndarray:
    result = RotateAndRescale(angle_in_degrees=angle_in_degrees).apply(RGBImage(image))
    return result.value


@gmt.relation(rotation_90deg_clockwise)
def rotation_90deg_clockwise_rel(x: int, y: int) -> bool:
    mapping: Dict[int, int] = rot_90_clockwise_relation_mapping
    x_hat: int = mapping.get(x, x)
    return gmt.equality(x_hat, y)


@gmt.valid_input(rotation_90deg_clockwise)
def rotation_90deg_clockwise_input(y: int) -> bool:
    valids = rot_90_clockwise_relation_mapping.keys()
    return y in valids


@gmt.transformation(rotation_90deg_counterclockwise)
@gmt.randomized("angle_in_degrees", gmt.RandInt(-100, -80))
def rotate_90deg_clockwise(image: np.ndarray, angle_in_degrees: int) -> np.ndarray:
    result = RotateAndRescale(angle_in_degrees=angle_in_degrees).apply(RGBImage(image))
    return result.value


@gmt.relation(rotation_90deg_counterclockwise)
def rotation_90deg_counterclockwise_rel(x: int, y: int) -> bool:
    mapping: Dict[int, int] = rot_90_counterclockwise_relation_mapping
    x_hat: int = mapping.get(x, x)
    return gmt.equality(x_hat, y)


@gmt.valid_input(rotation_90deg_counterclockwise)
def rotation_90deg_counterclockwise_input(y: int) -> bool:
    valids = rot_90_counterclockwise_relation_mapping.keys()
    return y in valids


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
