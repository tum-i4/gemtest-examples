from typing import List

import numpy as np

import gemtest as gmt
from examples.gtsrb.metamorphic_tests.setup_image_classifier import test_image_paths, \
    classifier_under_test, export_data, traffic_sign_visualizer
from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.gtsrb_label_enum import GTSRBLabelEnum as GTSRB
from examples.gtsrb.utils.triangle_sign.triangle_transformation import TriangleMorph

number_of_test_cases = 30

'''combine two triangular signs to create a new variation of the images'''
create_triangle = gmt.create_metamorphic_relation(
    name="create_triangle",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.SAMPLE,
    number_of_test_cases=number_of_test_cases,
    number_of_sources=2
)

create_triangle_2 = gmt.create_metamorphic_relation(
    name="create_triangle_2",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.SAMPLE,
    number_of_test_cases=number_of_test_cases,
    number_of_sources=2
)


def fuse_triangles_wrapper(background: np.ndarray, foreground: np.ndarray) -> RGBImage:
    background_length = min(background[:, :, 0].shape)
    foreground_length = min(foreground[:, :, 0].shape)
    if background_length < 45 and foreground_length < 45:
        gmt.skip("Sign creation failed")

    rgb_background = RGBImage(background)
    rgb_foreground = RGBImage(foreground)
    return TriangleMorph(rgb_background).apply(rgb_foreground)


@gmt.general_transformation(create_triangle)
def fuse_triangles(mtc: gmt.MetamorphicTestCase) -> List[np.ndarray]:
    img1 = mtc.source_inputs[0]
    img2 = mtc.source_inputs[1]
    result = fuse_triangles_wrapper(background=img2, foreground=img1)
    return result.value


@gmt.general_transformation(create_triangle_2)
def fuse_triangles(mtc: gmt.MetamorphicTestCase) -> List[np.ndarray]:
    img1 = mtc.source_inputs[0]
    img2 = mtc.source_inputs[1]
    result = fuse_triangles_wrapper(background=img1, foreground=img2)
    return result.value


triangle_keys = [
    GTSRB.PRIORITY_IN_TRAFFIC_NEXT_CROSSING,
    GTSRB.WARNING,
    GTSRB.DANGEROUS_CURVE_TO_LEFT,
    GTSRB.DANGEROUS_CURVE_TO_RIGHT,
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
    GTSRB.DEER_CROSSING
]


@gmt.valid_input(create_triangle, create_triangle_2)
def fuse_triangles_input(y: int) -> bool:
    return y in triangle_keys


@gmt.general_relation(create_triangle)
def is_swapped_triangle_sign(mtc: gmt.MetamorphicTestCase) -> bool:
    return mtc.source_outputs[0] == mtc.followup_output


@gmt.general_relation(create_triangle_2)
def is_swapped_triangle_sign(mtc: gmt.MetamorphicTestCase) -> bool:
    return mtc.source_outputs[1] == mtc.followup_output


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
