from typing import List

import albumentations
import cv2
import numpy as np

import gemtest as gmt
from examples.gtsrb.metamorphic_tests.setup_image_classifier import test_image_paths, traffic_sign_visualizer, \
    classifier_under_test, export_data

number_of_test_cases = 10
'''Only contains relations that do not change the label of the image.'''

'''Changes the brightness level of an image.'''
brightness = gmt.create_metamorphic_relation(
    name="brightness",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Changes the contrast of an image.'''
contrast = gmt.create_metamorphic_relation(
    name="contrast",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Changes both the brightness and contrast of an image. Makes use of the albumentations library to do so'''
both_transform = gmt.create_metamorphic_relation(
    name="both_transform",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Changes both the brightness and contrast of an image. Makes use of the cv2(OpenCV) library to do so.'''
both_cv2 = gmt.create_metamorphic_relation(
    name="both_cv2",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Creates artificial rain in the image by inserting raindrops.'''
rain = gmt.create_metamorphic_relation(
    name="rain",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Bleach out some pixel values simulating snow.'''
snow = gmt.create_metamorphic_relation(
    name="snow",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Creates artificial fog in the image.'''
fog = gmt.create_metamorphic_relation(
    name="fog",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Changes the gamma in the image.'''
gamma = gmt.create_metamorphic_relation(
    name="gamma",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Equalizes the image histogram.'''
equalize = gmt.create_metamorphic_relation(
    name="equalize",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Downscales the image.'''
downscale = gmt.create_metamorphic_relation(
    name="downscale",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Apply camera sensor noise.(ISO Noise)'''
noise = gmt.create_metamorphic_relation(
    name="noise",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the input image.'''
clahe = gmt.create_metamorphic_relation(
    name="clahe",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''Blur the input image using a random-sized kernel.'''
blur = gmt.create_metamorphic_relation(
    name="blur",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''CoarseDropout of the rectangular regions in the image.'''
dropout = gmt.create_metamorphic_relation(
    name="dropout",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)

'''A composition of CoarseDropout, Downscale and CLAHE.'''
trio = gmt.create_metamorphic_relation(
    name="drop_down_bright",
    data=test_image_paths,
    relation=gmt.equality,
    testing_strategy=gmt.TestingStrategy.EXHAUSTIVE,
    number_of_test_cases=number_of_test_cases
)


@gmt.transformation(brightness)
@gmt.randomized("beta", gmt.RandFloat(0.3, 1.7))
def brightness_adjustments(image: np.ndarray, beta: int) -> np.ndarray:
    image = image.astype(np.float32)
    return np.clip(beta * image, 0, 255).astype(np.uint8)


@gmt.transformation(contrast)
@gmt.randomized("alpha", gmt.RandFloat(0.6, 1.5))
def contrast_adjustments(image: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)


@gmt.transformation(both_transform)
@gmt.randomized("alpha", gmt.RandFloat(0.6, 1.5))
@gmt.randomized("beta", gmt.RandFloat(-0.7, 0.7))
def brightness_contrast_adjustments(image: np.ndarray, alpha: float, beta: int) -> np.ndarray:
    image = image.astype(np.float32)
    brightness_val = np.clip(image + beta * image, 0, 255)
    return np.clip(alpha * brightness_val, 0, 255).astype(np.uint8)


@gmt.transformation(both_cv2)
@gmt.randomized("alpha", gmt.RandFloat(0.6, 1.5))
@gmt.randomized("beta", gmt.RandInt(-1, 1))
def cv2_brightness_contrast_adjustments(
        image: np.ndarray, alpha: float, beta: int
) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


@gmt.transformation(rain)
@gmt.randomized("slant", gmt.RandInt(-5, 5))
def album_rain(
        image: np.ndarray,
        slant: int = 0,
        drop_length: int = 9,
        drop_width: int = 1,
        blur_value: int = 3,
) -> np.ndarray:
    image_transform = albumentations.RandomRain(
        slant_lower=slant,
        slant_upper=slant,
        drop_length=drop_length,
        drop_width=drop_width,
        blur_value=blur_value,
        p=1,
    )
    return image_transform.apply(image)


@gmt.transformation(snow)
@gmt.randomized("snow_point", gmt.RandFloat(0.1, 0.2))
def album_snow(
        image: np.ndarray, snow_point: float = 0.2, brightness_coef: float = 2
) -> np.ndarray:
    image_transform = albumentations.RandomSnow(
        snow_point_lower=snow_point,
        snow_point_upper=snow_point,
        brightness_coeff=brightness_coef,
        p=1,
    )
    return image_transform.apply(image)


@gmt.transformation(fog)
@gmt.randomized("fog_coef", gmt.RandFloat(0.3, 0.5))
def album_fog(
        image: np.ndarray, fog_coef: float = 0.5, alpha_coef: float = 0.08
) -> np.ndarray:
    image_transform = albumentations.RandomFog(
        fog_coef_lower=fog_coef, fog_coef_upper=fog_coef, alpha_coef=alpha_coef, p=1
    )
    return image_transform.apply(image)


@gmt.transformation(gamma)
@gmt.randomized("limit", gmt.RandInt(70, 130))
def album_gamma(image: np.ndarray, limit: int = 101) -> np.ndarray:
    # some transform need a little different setup
    image_transform = albumentations.Compose(
        [albumentations.RandomGamma(gamma_limit=(limit, limit), p=1)]
    )
    return image_transform(image=image)["image"]


@gmt.transformation(equalize)
def album_equalize(image: np.ndarray) -> np.ndarray:
    image_transform = albumentations.Equalize(p=1)
    return image_transform.apply(image)


@gmt.transformation(dropout)
@gmt.randomized("holes", gmt.RandInt(4, 6))
def album_dropout(
        image: np.ndarray, holes: int = 6, height: int = 6, width: int = 6
) -> np.ndarray:
    # some transform need a little different setup
    image_transform = albumentations.Compose(
        [
            albumentations.CoarseDropout(
                max_holes=holes, max_height=height, max_width=width, p=1
            )
        ]
    )
    return image_transform(image=image)["image"]


@gmt.transformation(downscale)
@gmt.randomized("scale", gmt.RandFloat(0.5, 0.7))
def album_downscale(image: np.ndarray, scale: float = 0.5) -> np.ndarray:
    image_transform = albumentations.Downscale(
        interpolation=albumentations.Downscale.Interpolation(downscale=cv2.INTER_NEAREST, upscale=cv2.INTER_NEAREST),
        p=1)
    return image_transform.apply(image, scale=scale)


@gmt.transformation(clahe)
@gmt.randomized("clip_limit", gmt.RandFloat(3.0, 3.5))
def album_clahe(
        image: np.ndarray, clip_limit: float = 3.0, tile_grid_size: int = 8
) -> np.ndarray:
    image_transform = albumentations.CLAHE(
        clip_limit=(clip_limit, clip_limit),
        tile_grid_size=(tile_grid_size, tile_grid_size),
        p=1,
    )
    return image_transform.apply(image)


@gmt.transformation(trio)
@gmt.randomized("holes", gmt.RandInt(4, 6))
@gmt.randomized("scale", gmt.RandFloat(0.5, 0.7))
@gmt.randomized("clip_limit", gmt.RandFloat(3.0, 3.5))
def album_trio(image: np.ndarray,
               scale: float,
               holes: int = 6, height: int = 6, width: int = 6,
               clip_limit: float = 3.0, tile_grid_size: int = 8
               ) -> np.ndarray:
    # dropout
    image_transform = albumentations.Compose(
        [
            albumentations.CoarseDropout(
                max_holes=holes, max_height=height, max_width=width, p=1
            )
        ]
    )
    dropout_img = image_transform(image=image)["image"]

    # downscale
    image_transform = albumentations.Downscale(
        interpolation=albumentations.Downscale.Interpolation(downscale=cv2.INTER_NEAREST, upscale=cv2.INTER_NEAREST),
        p=1)
    downscale_img = image_transform.apply(dropout_img, scale=scale)

    # clahe
    image_transform = albumentations.CLAHE(
        clip_limit=(clip_limit, clip_limit),
        tile_grid_size=(tile_grid_size, tile_grid_size),
        p=1,
    )
    return image_transform.apply(downscale_img)


@gmt.transformation(noise)
@gmt.randomized("color_shift", gmt.RandFloat(0.02, 0.04))
def album_iso_noise(
        image: np.ndarray, color_shift: float = 0.03, intensity: float = 0.3
) -> np.ndarray:
    image_transform = albumentations.ISONoise(
        color_shift=(color_shift, color_shift), intensity=(intensity, intensity), p=1
    )
    return image_transform.apply(image)


@gmt.transformation(blur)
@gmt.randomized("kernel_size", gmt.RandInt(3, 5))
def album_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    image_transform = albumentations.Blur(blur_limit=(kernel_size, kernel_size), p=1)
    return image_transform.apply(image)


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
