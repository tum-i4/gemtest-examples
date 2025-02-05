import numpy as np
import tensorflow as tf

import examples.svhn.data.utils as image_loader
import gemtest as gmt

tf2 = tf.compat.v2
from examples.svhn.metamorphic_tests.setup_svhn_classifier import test_image_paths, svhn_visualizer, \
    classifier_under_test
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imgaug.augmenters as iaa

number_of_test_cases = 10
img_channels = 3

image_rotate = gmt.create_metamorphic_relation(
    name="image_rotate_relation",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_shift = gmt.create_metamorphic_relation(
    name="image_translation_relation",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_scale = gmt.create_metamorphic_relation(
    name="image_scale_relation",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_erase_random = gmt.create_metamorphic_relation(
    name="image_erase_random_relation",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_erase_grid_random = gmt.create_metamorphic_relation(
    name="image_erase_grid_random_relation",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

image_shear = gmt.create_metamorphic_relation(
    name="image_shear_relation",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

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

image_shift_rotate = gmt.create_metamorphic_relation(
    name="image_shift_rotate_relation",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

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

brightness = gmt.create_metamorphic_relation(
    name="brightness",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

sharpness = gmt.create_metamorphic_relation(
    name="sharpness",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

posterize = gmt.create_metamorphic_relation(
    name="posterize",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

solarize = gmt.create_metamorphic_relation(
    name="solarize",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

contrast = gmt.create_metamorphic_relation(
    name="contrast",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)

histogram_equalize = gmt.create_metamorphic_relation(
    name="histogram_equalize",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases)


@gmt.transformation(solarize)
@gmt.randomized("threshold", gmt.RandFloat(0, 255))
def image_solarize(image: np.ndarray, threshold: int) -> np.ndarray:
    # solarize  the all pixel  above threshold
    seq = iaa.Sequential([iaa.Solarize(threshold=threshold)])
    return seq(image=image)


@gmt.transformation(brightness)
@gmt.randomized("mul", gmt.RandFloat(0.7, 1.3))
def image_brightness(image: np.ndarray, mul: float) -> np.ndarray:
    # mul: multiplication factor for brightness, 0.7 means darker, 1.3 means brighter image
    seq = iaa.Sequential([iaa.MultiplyBrightness(mul=mul)])
    return seq(image=image)


@gmt.transformation(sharpness)
@gmt.randomized("alpha", gmt.RandFloat(0, 1))
@gmt.randomized("lightness", gmt.RandFloat(0.75, 2.0))
def image_sharpness(image: np.ndarray, alpha: float, lightness: float) -> np.ndarray:
    # alpha: blending factor of sharpness , 0 = original image , 1= sharpened version
    # lightness: brightness of sharpened image 0.75=darker image , 2.0=brighter
    seq = iaa.Sequential([iaa.Sharpen(alpha=alpha, lightness=lightness)])
    return seq(image=image)


@gmt.transformation(posterize)
@gmt.randomized("nb_bits", gmt.RandInt(1, 8))
def image_posterize(image: np.ndarray, nb_bits: int) -> np.ndarray:
    # nb_bits parameter allows for different levels of posterization,
    # with lower values resulting in fewer colors and a more pronounced posterization effect.
    seq = iaa.Sequential([iaa.UniformColorQuantizationToNBits(nb_bits=nb_bits)])
    return seq(image=image)


@gmt.transformation(contrast)
@gmt.randomized("alpha", gmt.RandFloat(0.6, 1.4))
def image_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    seq = iaa.Sequential([iaa.LinearContrast(alpha=alpha)])
    return seq(image=image)


@gmt.transformation(histogram_equalize)
def image_histogram_equalize(image: np.ndarray) -> np.ndarray:
    seq = iaa.Sequential([iaa.HistogramEqualization()])
    return seq(image=image)


@gmt.transformation(image_rotate)
def image_rotate_transform(image: np.ndarray) -> np.ndarray:
    angle_range = [-30, 30]

    seq = iaa.Sequential([iaa.Rotate((angle_range[0], angle_range[1]))])
    aug_image = seq(image=image)
    return aug_image


@gmt.transformation(image_shift)
def image_shift_transform(image: np.ndarray) -> np.ndarray:
    left, right, top, bot = get_margins(image)
    seq = iaa.Sequential([
        iaa.Affine(translate_px={"x": (-left, right), "y": (-top, bot)})])
    aug_image = seq(image=image)
    return aug_image


@gmt.transformation(image_scale)
def image_scale_transform(image: np.ndarray) -> np.ndarray:
    width_range = [0.75, 1.25]
    height_range = [0.75, 1.25]
    seq = iaa.Sequential([
        iaa.Affine(scale={"x": (width_range[0], width_range[1]), "y": (height_range[0], height_range[1])})])
    aug_image = seq(image=image)
    return aug_image


# erases p percentage of the pixels
@gmt.transformation(image_erase_random)
def image_erase_random_transform(image: np.ndarray) -> np.ndarray:
    # iaa.CoarseDropout(p=0.0012755, size_percent=0.25)
    seq = iaa.Sequential([
        iaa.Dropout(p=(0, 0.2))])
    aug_image = seq(image=image)
    return aug_image


# cut a random 4x4 grid in the middle 20x20 part
@gmt.transformation(image_erase_grid_random)
def image_erase_grid_random_transform(image: np.ndarray) -> np.ndarray:
    # iaa.CoarseDropout(p=0.04, size_percent=0.25)
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


@gmt.transformation(image_shear)
def image_shear_transform(image: np.ndarray) -> np.ndarray:
    # ENSNet proposed -0.3 to 0.3 angle
    width_range = [-20, 20]
    height_range = [-20, 20]
    seq = iaa.Sequential([
        iaa.ShearX((width_range[0], width_range[1])),
        iaa.ShearY((height_range[0], height_range[1]))])
    aug_image = seq(image=image)
    return aug_image


@gmt.transformation(image_homography)
def image_homography_transform(image: np.ndarray) -> np.ndarray:
    # scale is roughly a measure of how far the perspective transformation’s corner points may be distanced from the image’s corner points
    seq = iaa.Sequential([iaa.PerspectiveTransform(scale=(0.01, 0.15))])
    aug_image = seq(image=image)
    return aug_image


@gmt.transformation(image_elastic)
def image_elastic_transform(image: np.ndarray) -> np.ndarray:
    # imgaug doc suggests A relation of 10:1. alpha=(0, 70.0), sigma=(4.0, 6.0) may be a good choice and will lead to a water-like effect.
    # but APAC paper suggests alpha=8, sigma=6 for mnist
    seq = iaa.Sequential([iaa.ElasticTransformation(alpha=8, sigma=6)])
    aug_image = seq(image=image)
    return aug_image


@gmt.transformation(image_shift_rotate)
def image_shift_rotate_transform(image: np.ndarray) -> np.ndarray:
    angle_range = [-30, 30]
    left, right, top, bot = get_margins(image)
    seq = iaa.Sequential([
        iaa.Affine(translate_px={"x": (-left, right), "y": (-top, bot)}),
        iaa.Rotate((angle_range[0], angle_range[1]))])
    aug_image = seq(image=image)

    return aug_image


@gmt.transformation(image_180_rotate)
def image_180_rotate_transform(image: np.ndarray) -> np.ndarray:
    seq = iaa.Sequential([iaa.Rot90(2)])
    aug_image = seq(image=image)
    return aug_image


@gmt.transformation(image_horizontal_flip)
def image_horizontal_flip_transform(image: np.ndarray) -> np.ndarray:
    seq = iaa.Sequential([iaa.Fliplr()])
    aug_image = seq(image=image)
    return aug_image


@gmt.relation(image_horizontal_flip)
def image_horizontal_flip_transform(source_output: int, followup_output: int):
    return gmt.equality(source_output, followup_output)


@gmt.relation(image_180_rotate)
def image_180_rotate_relation(source_output: int, followup_output: int):
    return gmt.equality(source_output, followup_output)


@gmt.valid_input(image_180_rotate, image_horizontal_flip)
def image_180_rotate_input(y: int) -> bool:
    valids = [8, 0]
    return y in valids


def get_margins(image):
    rows, cols, channels = image.shape
    img_size = rows  # Assuming a square image
    image = tf.reshape(image, [img_size, img_size, channels])
    nonzero_x_cols = tf.cast(tf.where(tf.greater(
        tf.reduce_sum(image, axis=0), 0)), tf.int32)
    nonzero_y_rows = tf.cast(tf.where(tf.greater(
        tf.reduce_sum(image, axis=1), 0)), tf.int32)
    left_margin = tf.reduce_min(nonzero_x_cols)
    right_margin = img_size - tf.reduce_max(nonzero_x_cols) - 1
    top_margin = tf.reduce_min(nonzero_y_rows)
    bot_margin = img_size - tf.reduce_max(nonzero_y_rows) - 1
    return left_margin.numpy(), right_margin.numpy(), top_margin.numpy(), bot_margin.numpy()


@gmt.system_under_test(
    visualize_input=svhn_visualizer.visualize_input,
    visualize_output=svhn_visualizer.visualize_output,
    data_loader=image_loader.load_image_resource,
    # data_exporter=export_data
)
def test_image_classifier(image: np.ndarray) -> int:
    """Predict the traffic sign in an image"""
    return classifier_under_test.evaluate_image(image)
