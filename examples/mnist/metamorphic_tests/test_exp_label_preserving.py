import imgaug.augmenters as iaa  # https://imgaug.readthedocs.io/en/latest/
import numpy as np

import gemtest as gmt
import examples.mnist.data.data_loaders as image_loader
from examples.mnist.libs.MorphoMNIST.morphomnist import morpho, perturb
from examples.mnist.metamorphic_tests.setup_torch_classifier import (test_image_paths, mnist_visualizer,
                                                                     classifier_under_test, export_data)

number_of_test_cases = 10
img_channels = 3

image_swelling = gmt.create_metamorphic_relation(
    name="image_swelling",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_fracture = gmt.create_metamorphic_relation(
    name="image_fracture",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_brightness = gmt.create_metamorphic_relation(
    name="image_brightness",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_contrast = gmt.create_metamorphic_relation(
    name="image_contrast",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_gaussianBlur = gmt.create_metamorphic_relation(
    name="image_gaussianBlur",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)


@gmt.transformation(image_swelling)
def image_swelling_transform(image: np.ndarray) -> np.ndarray:
    # problem: a 1 can get a 9 or 6
    # out = np.array(_line_thickening_thinning(image), dtype=image.dtype)
    morphology = morpho.ImageMorphology(np.squeeze(image))
    out = perturb.Swelling(strength=3, radius=7)(morphology)
    out = morphology.downscale(out)
    out = np.expand_dims(out, -1)
    return out


@gmt.transformation(image_fracture)
def image_fracture_transform(image: np.ndarray) -> np.ndarray:
    morphology = morpho.ImageMorphology(np.squeeze(image))
    out = perturb.Fracture(num_frac=3)(morphology)
    out = morphology.downscale(out)
    out = np.expand_dims(out, -1)
    return out


@gmt.transformation(image_brightness)
def image_brightness_transform(image: np.ndarray) -> np.ndarray:
    # Change brightness of images (50-150% of original value).
    seq = iaa.Sequential([iaa.Multiply((0.5, 1.5))])
    aug_image = seq(image=image)

    return aug_image


@gmt.transformation(image_contrast)
def image_brightness_transform(image: np.ndarray) -> np.ndarray:
    seq = iaa.Sequential([iaa.LinearContrast((0.5, 2.0))])
    aug_image = seq(image=image)

    return aug_image


@gmt.transformation(image_gaussianBlur)
def image_gaussianBlur_transform(image: np.ndarray) -> np.ndarray:
    seq = iaa.Sequential([iaa.GaussianBlur(sigma=(0.0, 3.0))])
    aug_image = seq(image=image)

    return aug_image


@gmt.system_under_test(
    visualize_input=mnist_visualizer.visualize_input,
    visualize_output=mnist_visualizer.visualize_output,
    data_loader=image_loader.load_image_resource,
    data_exporter=export_data
)
def test_image_classifier(image: np.ndarray) -> int:
    """Predict the traffic sign in an image"""
    return classifier_under_test.evaluate_image(image)
