import os
import sys
from typing import List

import numpy as np

import gemtest as gmt
from examples.gtsrb.metamorphic_tests.setup_image_classifier import test_image_paths, traffic_sign_visualizer, \
    classifier_under_test, export_data
from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.processing_steps.overlay_operation import OverlayOperation

number_of_test_cases = 10

module_folder = os.path.dirname(__file__)
one_folders_up = os.path.dirname(module_folder)

'''Overlays the image with an image of a branch.'''
overlay_branch = gmt.create_metamorphic_relation(
    name="overlay_branch",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''Overlays the image with an image of a building.'''
overlay_building = gmt.create_metamorphic_relation(
    name="overlay_building",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''Overlays the image with an image of a pole.'''
overlay_pole = gmt.create_metamorphic_relation(
    name="overlay_pole",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''Overlays the image with one or two stickers. 
This depends on how much space is left after applying the first sticker.'''
overlay_sticker = gmt.create_metamorphic_relation(
    name="overlay_sticker",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''Overlays the image with an image of vehicle.'''
overlay_vehicle = gmt.create_metamorphic_relation(
    name="overlay_vehicle",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)


@gmt.transformation(overlay_branch)
@gmt.randomized("image_number", gmt.RandInt(0, int(sys.maxsize)))
@gmt.randomized("rotation_degrees", gmt.RandInt(0, 360))
@gmt.randomized("scaling_factor", gmt.RandFloat(0.4, 1.0))
def overlay_branch(image: np.ndarray,
                   image_number: int,
                   rotation_degrees: int,
                   scaling_factor: float
                   ) -> np.ndarray:
    op = OverlayOperation(
        overlay_images_dir=os.path.join(one_folders_up, "assets/overlay/branches"),
        # between 20% and 70% of road sign covered with overlaid image
        min_nontransparent_ratio=0.2,
        max_nontransparent_ratio=0.7,
        # randomized variables
        overlay_image_number=image_number,
        rotation_degrees=rotation_degrees,
        scaling_factor=scaling_factor,
    )
    img = RGBImage(image)
    result = op.apply(img)
    if result is None:
        gmt.skip("The transformation overlay_branch() failed")
    return result.value


@gmt.transformation(overlay_building)
@gmt.randomized("image_number", gmt.RandInt(0, int(sys.maxsize)))
@gmt.randomized("scaling_factor", gmt.RandFloat(0.7, 2.5))
def overlay_building(image: np.ndarray,
                     image_number: int,
                     scaling_factor: float
                     ) -> np.ndarray:
    op = OverlayOperation(
        overlay_images_dir=os.path.join(one_folders_up, "assets/overlay/buildings"),
        # between 20% and 70% of road sign covered with overlaid image
        min_nontransparent_ratio=0.2,
        max_nontransparent_ratio=0.7,
        # ensures buildings are not rotated or lifted off the ground
        rotate=False,
        min_nontransparent_ratio_bottom=0.3,
        # randomized variables
        overlay_image_number=image_number,
        scaling_factor=scaling_factor,
    )
    img = RGBImage(image)
    result = op.apply(img)
    if result is None:
        gmt.skip("The transformation overlay_building() failed")
    return result.value


@gmt.transformation(overlay_pole)
@gmt.randomized("image_number", gmt.RandInt(0, int(sys.maxsize)))
@gmt.randomized("scaling_factor", gmt.RandFloat(1.0, 2.5))
def overlay_pole(image: np.ndarray,
                 image_number: int,
                 scaling_factor: float
                 ) -> np.ndarray:
    op = OverlayOperation(
        overlay_images_dir=os.path.join(one_folders_up, "assets/overlay/poles"),
        # between 5% and 70% of road sign covered with overlaid image
        min_nontransparent_ratio=0.05,
        max_nontransparent_ratio=0.7,
        # ensures poles are not rotated or lifted off the ground
        rotate=False,
        min_nontransparent_ratio_bottom=0.001,
        # randomized variables
        overlay_image_number=image_number,
        scaling_factor=scaling_factor,
    )
    img = RGBImage(image)
    result = op.apply(img)
    if result is None:
        gmt.skip("The transformation overlay_pole() failed")
    return result.value


@gmt.transformation(overlay_sticker)
@gmt.randomized("image_number", gmt.RandInt(0, int(sys.maxsize)))
@gmt.randomized("rotation_degrees1", gmt.RandInt(0, 360))
@gmt.randomized("rotation_degrees2", gmt.RandInt(0, 360))
@gmt.randomized("scaling_factor1", gmt.RandFloat(0.2, 0.8))
@gmt.randomized("scaling_factor2", gmt.RandFloat(0.15, 0.5))
def overlay_sticker(image: np.ndarray,
                    image_number: int,
                    rotation_degrees1: int,
                    rotation_degrees2: int,
                    scaling_factor1: float,
                    scaling_factor2: float,
                    ) -> np.ndarray:
    op1 = OverlayOperation(
        overlay_images_dir=os.path.join(one_folders_up, "assets/overlay/stickers"),
        # between 3% and 70% of road sign covered with first sticker
        min_nontransparent_ratio=0.03,
        max_nontransparent_ratio=0.7,
        # randomized variables
        overlay_image_number=image_number,
        rotation_degrees=rotation_degrees1,
        scaling_factor=scaling_factor1,
    )
    op2 = OverlayOperation(
        overlay_images_dir=os.path.join(one_folders_up, "assets/overlay/stickers"),
        # between 1% and 30% of road sign covered with second sticker
        min_nontransparent_ratio=0.01,
        max_nontransparent_ratio=0.3,
        # randomized variables
        overlay_image_number=image_number,
        rotation_degrees=rotation_degrees2,
        scaling_factor=scaling_factor2,
    )
    img = RGBImage(image)
    result = op1.apply(img)
    if result is None:
        gmt.skip("The first transformation overlay_sticker() failed")
    result = op2.apply(result)
    if result is None:
        gmt.skip("The second transformation overlay_sticker() failed")
    return result.value


@gmt.transformation(overlay_vehicle)
@gmt.randomized("image_number", gmt.RandInt(0, int(sys.maxsize)))
@gmt.randomized("scaling_factor", gmt.RandFloat(0.5, 1.2))
def overlay_vehicle(
        image: np.ndarray,
        image_number: int,
        scaling_factor: float
) -> np.ndarray:
    op = OverlayOperation(
        overlay_images_dir=os.path.join(one_folders_up, "assets/overlay/vehicles"),
        # between 20% and 70% of road sign covered with overlaid image
        min_nontransparent_ratio=0.2,
        max_nontransparent_ratio=0.7,
        # ensures vehicles are not rotated or lifted off the ground
        rotate=False,
        min_nontransparent_ratio_bottom=0.1,
        # randomized variables
        overlay_image_number=image_number,
        scaling_factor=scaling_factor,
    )
    img = RGBImage(image)
    result = op.apply(img)
    if result is None:
        gmt.skip("The transformation overlay_vehicle() failed")
    return result.value


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
