import random

import cv2
import imgaug.augmenters as iaa  # https://imgaug.readthedocs.io/en/latest/
import numpy as np
import tensorflow as tf

import examples.mnist.data.data_loaders as image_loader
import examples.mnist.utils.augmentationUtils as utils
import gemtest as gmt
from examples.mnist.libs.MorphoMNIST.morphomnist import morpho, \
    perturb  # https://github.com/dccastro/Morpho-MNIST/tree/master
from examples.mnist.metamorphic_tests.setup_torch_classifier import (test_image_paths, mnist_visualizer,
                                                                     classifier_under_test, export_data)

tf2 = tf.compat.v2

number_of_test_cases = 10
scale_width_range = [0.75, 1.25]
scale_height_range = [0.75, 1.25]

image_rotate = gmt.create_metamorphic_relation(
    name="image_rotate",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_shift = gmt.create_metamorphic_relation(
    name="image_shift",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_scale = gmt.create_metamorphic_relation(
    name="image_scale",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_erase_random = gmt.create_metamorphic_relation(
    name="image_erase_random",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_erase_grid_random = gmt.create_metamorphic_relation(
    name="image_erase_grid_random",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_shear = gmt.create_metamorphic_relation(
    name="image_shear",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases,
    number_of_sources=1)

image_homography = gmt.create_metamorphic_relation(
    name="image_homography_transform",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_elastic = gmt.create_metamorphic_relation(
    name="image_elastic_transform",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_line_thickening = gmt.create_metamorphic_relation(
    name="image_line_thickening",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_line_thickening2 = gmt.create_metamorphic_relation(
    name="image_line_thickening2",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_line_thinning = gmt.create_metamorphic_relation(
    name="image_line_thinning",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_line_thinning2 = gmt.create_metamorphic_relation(
    name="image_line_thinning2",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_shift_rotate = gmt.create_metamorphic_relation(
    name="image_shift&rotate",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_hem = gmt.create_metamorphic_relation(
    name="image_hem",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''applied on 0&8 only'''
image_vertical_flip = gmt.create_metamorphic_relation(
    name="image_vertical_flip",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

'''applied on 0&8 only'''
image_horizontal_flip = gmt.create_metamorphic_relation(
    name="image_horizontal_flip",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_scale_shift = gmt.create_metamorphic_relation(
    name="image_scale_shift_relation",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

# applied on 0&8 only
image_vertical_flip_shift = gmt.create_metamorphic_relation(
    name="image_vertical_flip&shift",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

# applied on 0&8 only
image_horizontal_flip_shift = gmt.create_metamorphic_relation(
    name="image_horizontal_flip&shift",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)


@gmt.transformation(image_rotate)
def image_rotate_transform(image: np.ndarray) -> np.ndarray:
    angle_range = [-30, 30]

    seq = iaa.Sequential([iaa.Rotate((angle_range[0], angle_range[1]))])
    aug_image = seq(image=image)
    return aug_image


@gmt.transformation(image_shift)
def image_shift_transform(image: np.ndarray) -> np.ndarray:
    left, right, top, bot = utils.get_margins(image)
    aug_image = utils.shift(image, (-left, right), (-top, bot))
    return aug_image


@gmt.transformation(image_scale)
def image_scale_transform(image: np.ndarray) -> np.ndarray:
    aug_image = utils.scale(image, scale_width_range, scale_height_range)
    return aug_image


# erases p percentage of the pixels
@gmt.transformation(image_erase_random)
def image_erase_random_transform(image: np.ndarray) -> np.ndarray:
    seq = iaa.Sequential([
        iaa.Dropout(p=(0, 0.2))])
    aug_image = seq(image=image)
    return aug_image


# cut a random 4x4 grid in the middle 20x20 part
@gmt.transformation(image_erase_grid_random)
def image_erase_grid_random_transform(image: np.ndarray) -> np.ndarray:
    # Cut the 8-pixel border
    cut_border = 4
    cropped_image = image[cut_border:-cut_border, cut_border:-cut_border]
    seq = iaa.Sequential([
        iaa.Cutout(fill_mode="constant", cval=0, size=0.2, squared=True)])
    aug_image = seq(image=cropped_image)
    # Sticking the cropped image back to the original image
    stitched_image = np.copy(image)
    stitched_image[cut_border:-cut_border, cut_border:-cut_border] = aug_image
    return stitched_image


@gmt.general_transformation(image_shear)
def image_shear_transform(mtc: gmt.MetamorphicTestCase) -> np.ndarray:
    # took proposed values from the paper "Deep Big Simple Neural Nets Excel on Handwritten Digit Recognition"
    intensity = 15
    image = mtc.source_inputs[0]
    label = mtc.source_outputs[0]
    if label == 1 or label == 7:
        # take less intensity for 1 and 7 to reduce the risk of changing the label
        intensity = 7.5

    seq = iaa.Sequential([
        iaa.ShearX((-intensity, intensity)),
        iaa.ShearY((-intensity, intensity))])
    aug_image = seq(image=image)
    return aug_image


@gmt.transformation(image_homography)
def image_homography_transform(image: np.ndarray) -> np.ndarray:
    # scale is roughly a measure of how far the perspective transformation’s
    # corner points may be distanced from the image’s corner points
    seq = iaa.Sequential([iaa.PerspectiveTransform(scale=(0.01, 0.15))])
    aug_image = seq(image=image)
    return aug_image


@gmt.transformation(image_elastic)
def image_elastic_transform(image: np.ndarray) -> np.ndarray:
    # imgaug doc suggests A relation of 10:1. alpha=(0, 70.0), sigma=(4.0, 6.0) may
    # be a good choice and will lead to a water-like effect.
    # but APAC paper suggests alpha=8, sigma=6 for mnist
    seq = iaa.Sequential([iaa.ElasticTransformation(alpha=8, sigma=6)])
    aug_image = seq(image=image)
    return aug_image


@gmt.transformation(image_line_thickening)
def image_thickening_transform(image: np.ndarray) -> np.ndarray:
    random_amount = random.uniform(0.4, 0.8)
    morphology = morpho.ImageMorphology(np.squeeze(image))
    out = perturb.Thickening(amount=random_amount)(morphology)
    out = morphology.downscale(out)
    out = np.expand_dims(out, -1)
    return out


@gmt.transformation(image_line_thinning)
def image_thinning_transform(image: np.ndarray) -> np.ndarray:
    random_amount = random.uniform(0.3, 0.6)
    morphology = morpho.ImageMorphology(np.squeeze(image))
    out = perturb.Thinning(amount=random_amount)(morphology)
    out = morphology.downscale(out)
    out = np.expand_dims(out, -1)
    return out


@gmt.transformation(image_line_thinning2)
def image_thinning_transform2(image: np.ndarray) -> np.ndarray:
    out = np.array(_line_thinning_thickening(image, True), dtype=image.dtype)
    return out


@gmt.transformation(image_line_thickening2)
def image_thickening_transform2(image: np.ndarray) -> np.ndarray:
    out = np.array(_line_thinning_thickening(image, False), dtype=image.dtype)
    return out


@gmt.transformation(image_shift_rotate)
def image_shift_rotate_transform(image: np.ndarray) -> np.ndarray:
    angle_range = [-30, 30]
    left, right, top, bot = utils.get_margins(image)
    seq = iaa.Sequential([
        iaa.Affine(translate_px={"x": (-left, right), "y": (-top, bot)}),
        iaa.Rotate((angle_range[0], angle_range[1]))])
    aug_image = seq(image=image)

    return aug_image


@gmt.transformation(image_hem)
@gmt.randomized("coin", gmt.RandInt(0, 1))
def image_hem_transform(image: np.ndarray, coin) -> np.ndarray:
    seq = iaa.Sequential([iaa.PerspectiveTransform(scale=(0.01, 0.15)),
                          iaa.ElasticTransformation(alpha=38, sigma=6)])
    aug_image = seq(image=image)
    if coin == 0:
        out = np.array(_line_thinning_thickening(aug_image, True), dtype=image.dtype)
    else:
        out = np.array(_line_thinning_thickening(aug_image, False), dtype=image.dtype)
    return out


@gmt.transformation(image_vertical_flip)
def image_vertical_flip_transform(image: np.ndarray) -> np.ndarray:
    aug_image = utils.vertical_flip(image)
    return aug_image


@gmt.transformation(image_horizontal_flip)
def image_horizontal_flip_transform(image: np.ndarray) -> np.ndarray:
    aug_image = utils.horizontal_flip(image)
    return aug_image


@gmt.transformation(image_scale_shift)
def image_scale_shift_transform(image: np.ndarray) -> np.ndarray:
    left, right, top, bot = utils.get_margins(image)
    aug_image = utils.scale(image, scale_width_range, scale_height_range)
    aug_image = utils.shift(aug_image, (-left, right), (-top, bot))
    return aug_image


@gmt.transformation(image_vertical_flip_shift)
def image_vertical_flip_shift_transform(image: np.ndarray) -> np.ndarray:
    left, right, top, bot = utils.get_margins(image)
    aug_image = utils.vertical_flip(image)
    aug_image = utils.shift(aug_image, (-left, right), (-top, bot))
    return aug_image


@gmt.transformation(image_horizontal_flip_shift)
def image_horizontal_flip_shift_transform(image: np.ndarray) -> np.ndarray:
    left, right, top, bot = utils.get_margins(image)
    aug_image = utils.horizontal_flip(image)
    aug_image = utils.shift(aug_image, (-left, right), (-top, bot))
    return aug_image


@gmt.valid_input(image_vertical_flip, image_horizontal_flip, image_vertical_flip_shift, image_horizontal_flip_shift)
def image_vertical_flip_input(y: int) -> bool:
    valids = [8, 0]
    return y in valids


def _line_thinning_thickening(image, thinning: bool = True):
    # https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html
    image = np.squeeze(image)
    max_thickness = 2
    # Choose a random thickness for the lines
    thickness = np.random.randint(1, max_thickness + 1)
    # Apply random line thickening or thinning
    kernel = np.ones((thickness, thickness), dtype=np.uint8)
    if thinning:
        # The kernel slides through the image (as in 2D convolution).
        # A pixel in the original image (either 1 or 0) will be
        # considered 1 only if all the pixels under the kernel is 1,
        # otherwise it is eroded (made to zero).
        augmented_image = cv2.erode(image, kernel, iterations=1)
    else:
        # Here, a pixel element is '1' if at least one pixel under the kernel is '1'
        augmented_image = cv2.dilate(image, kernel, iterations=1)
    augmented_image = np.expand_dims(augmented_image, axis=-1)
    return augmented_image


@gmt.system_under_test(
    visualize_input=mnist_visualizer.visualize_input,
    visualize_output=mnist_visualizer.visualize_output,
    data_loader=image_loader.load_image_resource,
    data_exporter=export_data
)
def test_image_classifier(image: np.ndarray) -> int:
    """Predict the traffic sign in an image"""
    return classifier_under_test.evaluate_image(image)
