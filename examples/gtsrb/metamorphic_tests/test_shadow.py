from typing import List

import albumentations
import numpy as np

import gemtest as gmt
from examples.gtsrb.metamorphic_tests.setup_image_classifier import test_image_paths, \
    classifier_under_test, export_data, traffic_sign_visualizer
from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.shadow_transformation.shadow_tools import ShadowGenerationPipeline

number_of_test_cases = 10
'''Puts a partial shadow on top of the image. 
The shadow does not float in the image but starts and end at the image borders.'''
shadow = gmt.create_metamorphic_relation(
    name="shadow",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''Puts a shadow on top of the image. The shadow is created with albumentations.'''
albumentations_shadow = gmt.create_metamorphic_relation(
    name="albumentations_shadow",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)


@gmt.transformation(albumentations_shadow)
def album_shadow(image: np.ndarray) -> np.ndarray:
    image_transform = albumentations.RandomShadow(shadow_roi=(0, 0, 1, 1),
                                                  num_shadows_lower=1,
                                                  num_shadows_upper=3,
                                                  shadow_dimension=3,
                                                  p=1)
    return image_transform(image=image)["image"]


@gmt.transformation(shadow)
@gmt.randomized("shade", gmt.RandFloat(-1, 1))
def custom_shadow(image: np.ndarray, shade: float) -> np.ndarray:
    image_rgb = ShadowGenerationPipeline(shade).apply(RGBImage(image))
    return image_rgb.value


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
    batch_size=128
)
def test_mutant_image_classifier(images: List[np.ndarray], dynamic_sut) -> List[int]:
    with dynamic_sut:
        return dynamic_sut.execute(images)
