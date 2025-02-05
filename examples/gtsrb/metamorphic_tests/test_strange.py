from scipy import ndimage
import numpy as np

from typing import List

import albumentations
import numpy as np

import gemtest as gmt
from examples.gtsrb.utils.gtsrb_label_enum import GTSRBLabelEnum as GTSRB
from examples.gtsrb.metamorphic_tests.setup_image_classifier import test_image_paths, \
    traffic_sign_visualizer, export_data, classifier_under_test

number_of_test_cases = 10

"""
Contains all transformations which are extremely unlikely or impossible to occur in a real world setting. These
transformations are all summarized in this file to determine the effect of these transformations on the classifier
during training.
"""

# todo: D4 transformations for stop and empty symetrical signs
#  https://albumentations.ai/docs/api_reference/full_reference/?h=d4#albumentations.augmentations.geometric
#  .transforms.D4

'''Apply all 8 Symmetry relations from the D8 group.'''
d8_transform = gmt.create_metamorphic_relation(
    name="Dihedral Group D8 transform",
    data=test_image_paths,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Randomly Drop Channels in the input image.'''
channel_dropout = gmt.create_metamorphic_relation(
    name="channel_dropout",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Randomly rearrange channels of the input RGB image.'''
channel_shuffle = gmt.create_metamorphic_relation(
    name="channel_shuffle",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Randomly changes the brightness, contrast, and saturation of an image.'''
color_jitter = gmt.create_metamorphic_relation(
    name="color_jitter",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Apply defocus transform to the image.'''
defocus = gmt.create_metamorphic_relation(
    name="defocus",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Emboss the input image and overlays the result with the original image.'''
emboss = gmt.create_metamorphic_relation(
    name="emboss",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Apply glass noise to the input image.'''
glass_blur = gmt.create_metamorphic_relation(
    name="glass_blur",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Randomly change hue, saturation and value of the input image.'''
hue_saturation = gmt.create_metamorphic_relation(
    name="hue_saturation",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Invert the input image by subtracting pixel values from 255.'''
invert_color = gmt.create_metamorphic_relation(
    name="invert_color",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Augment RGB image using FancyPCA from Krizhevsky's paper 
"ImageNet Classification with Deep Convolutional Neural Networks"'''
pca = gmt.create_metamorphic_relation(
    name="pca",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Reduce the number of bits for each color channel.'''
posterize = gmt.create_metamorphic_relation(
    name="posterize",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Create ringing or overshoot artefacts by conlvolving image with 2D sinc filter.'''
ringing_overshoot = gmt.create_metamorphic_relation(
    name="ringing_overshoot",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Randomly shift values for each channel of the input RGB image.'''
rgb_shift = gmt.create_metamorphic_relation(
    name="rgb_shift",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Invert all pixel values above a threshold.'''
solarize = gmt.create_metamorphic_relation(
    name="solarize",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

D8_SIGNS = [
    GTSRB.STOP,
    GTSRB.NO_VEHICLES_PERMITTED_CIRCLE,
    GTSRB.ROUNDABOUT,
    GTSRB.PRIORITY_IN_TRAFFIC_ROAD,
]


# https://de.wikipedia.org/wiki/Diedergruppe
@gmt.general_transformation(d8_transform)
def d8_transformations(mtc: gmt.MetamorphicTestCase, d=8) -> List[np.ndarray]:
    img = mtc.source_inputs[0]
    follow_ups = list()
    c_in = 0.5 * np.array(img.shape)
    c_out = 0.5 * np.array(img.shape)
    a_k = [x * (np.pi / d) for x in range(d)]
    for i in range(d):
        transform_0 = np.array([[np.cos(a_k[i]), -np.sin(a_k[i]), 0],
                                [np.sin(a_k[i]), np.cos(a_k[i]), 0],
                                [0, 0, 1]])
        offset_0 = c_in - c_out.dot(transform_0)
        transform_1 = np.array([[np.cos(a_k[i]), np.sin(a_k[i]), 0],
                                [np.sin(a_k[i]), -np.cos(a_k[i]), 0],
                                [0, 0, 1]])
        offset_1 = c_in - c_out.dot(transform_1)
        follow_ups.append(ndimage.affine_transform(img, transform_0.T, offset=offset_0))
        follow_ups.append(ndimage.affine_transform(img, transform_1.T, offset=offset_1))
    return follow_ups


@gmt.valid_input(d8_transform)
def is_symmetric(y: int) -> bool:
    return y in D8_SIGNS


@gmt.general_relation(d8_transform)
def image_horizontal_flip_relation(mtc: gmt.MetamorphicTestCase) -> bool:
    source_output = mtc.source_outputs[0]
    return all(i == source_output for i in mtc.followup_outputs)


@gmt.transformation(posterize)
@gmt.randomized("bits", gmt.RandInt(2, 8))
def album_posterize(image: np.ndarray, bits: int = 7) -> np.ndarray:
    bits_array = [2, bits]
    image_transform = albumentations.Posterize(num_bits=[bits_array, bits_array, bits_array], p=1)
    transformed_image = image_transform(image=image)["image"]
    return transformed_image


@gmt.transformation(defocus)
@gmt.randomized("radius", gmt.RandInt(1, 5))
def album_defocus(image: np.ndarray, radius: int = 3) -> np.ndarray:
    image_transform = albumentations.Defocus(radius=(radius, radius), p=1)
    return image_transform(image=image)["image"]


@gmt.transformation(ringing_overshoot)
@gmt.randomized("blur_limit", gmt.RandInt(1, 10))
def album_ringing_overshoot(image: np.ndarray, blur_limit: int = 5) -> np.ndarray:
    blur_limit = blur_limit * 2 + 1
    image_transform = albumentations.RingingOvershoot(blur_limit=(blur_limit, blur_limit), p=1)
    return image_transform(image=image)["image"]


@gmt.transformation(channel_shuffle)
def album_channel_shuffle(image: np.ndarray) -> np.ndarray:
    image_transform = albumentations.ChannelShuffle(p=1)
    return image_transform(image=image)["image"]


@gmt.transformation(channel_dropout)
def album_channel_dropout(image: np.ndarray) -> np.ndarray:
    image_transform = albumentations.ChannelDropout(channel_drop_range=(1, 2), p=1)
    return image_transform(image=image)["image"]


@gmt.transformation(color_jitter)
@gmt.randomized("hue", gmt.RandFloat(-0.5, 0.5))
def album_color_jitter(image: np.ndarray, hue: float = 0.0) -> np.ndarray:
    image_transform = albumentations.ColorJitter(hue=(hue, hue), p=1)
    return image_transform(image=image)["image"]


@gmt.transformation(emboss)
@gmt.randomized("alpha", gmt.RandFloat(0.0, 1.0))
def album_emboss(image: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    image_transform = albumentations.Emboss(p=1.0, alpha=(alpha, alpha))
    return image_transform(image=image)["image"]


@gmt.transformation(pca)
@gmt.randomized("alpha", gmt.RandFloat(0.0, 1.5))
def album_fancy_pca(image: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    image_transform = albumentations.FancyPCA(p=1.0, alpha=alpha)
    return image_transform(image=image)["image"]


@gmt.transformation(glass_blur)
@gmt.randomized("sigma", gmt.RandFloat(0.0, 1.5))
@gmt.randomized("max_delta", gmt.RandInt(1, 4))
def album_glass_blur(image: np.ndarray, sigma: float = 0.0, max_delta: int = 1) -> np.ndarray:
    image_transform = albumentations.GlassBlur(p=1.0, sigma=sigma, max_delta=max_delta, iterations=2)
    return image_transform(image=image)["image"]


@gmt.transformation(hue_saturation)
def album_hue_saturation(image: np.ndarray) -> np.ndarray:
    image_transform = albumentations.HueSaturationValue(p=1.0)
    return image_transform(image=image)["image"]


@gmt.transformation(invert_color)
def album_invert_image(image: np.ndarray) -> np.ndarray:
    image_transform = albumentations.InvertImg(p=1.0)
    return image_transform(image=image)["image"]


@gmt.transformation(rgb_shift)
@gmt.randomized("r_shift", gmt.RandInt(-120, 120))
@gmt.randomized("g_shift", gmt.RandInt(-120, 120))
@gmt.randomized("b_shift", gmt.RandInt(-120, 120))
def album_rgb_shift(image: np.ndarray, r_shift: int = 0, g_shift: int = 0, b_shift: int = 0) -> np.ndarray:
    image_transform = albumentations.RGBShift(p=1.0, r_shift_limit=(r_shift, r_shift),
                                              g_shift_limit=(g_shift, g_shift),
                                              b_shift_limit=(b_shift, b_shift))
    return image_transform(image=image)["image"]


@gmt.transformation(solarize)
@gmt.randomized("threshold", gmt.RandInt(0, 255))
def album_solarize(image: np.ndarray, threshold: int = 150) -> np.ndarray:
    image_transform = albumentations.Solarize(p=1.0, threshold=(threshold, threshold))
    return image_transform(image=image)["image"]


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
    batch_size=64
)
def test_mutant_image_classifier(images: List[np.ndarray], dynamic_sut) -> List[int]:
    with dynamic_sut:
        return dynamic_sut.execute(images)
