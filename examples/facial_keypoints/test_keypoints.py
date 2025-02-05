from typing import List

import albumentations  # type: ignore
import cv2  # type: ignore
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

import gemtest as gmt
from .keypoint_detection import read_keypoint_images, KeypointModel
from .keypoint_visualizer import KeypointVisualizer

"""
This example demonstrates MR tests of a keypoint prediction NN on human portraits:
if the portraits are perturbed slightly, the prediction should not differ by much.

Be advised that the relation function makes use of this NN's training error function.
Other NN MR tests could apply the same principle, where the error between the first and
second output are small enough, using its own training error function.

What follows here are 9 image perturbation functions, all with the same signature: receives
an image, and one or more parameters that can be used on the perturbation function.
Consult albumentations documentation for more information on functions that uses them:
https://albumentations.ai/docs/api_reference/augmentations/transforms/.
Note that the randomization of some of the parameters are delegated into our own framework
instead of using the perturbation's own random functionality.
Also there are less perturbations that can be used here than the classifier example, as
there are perturbations that can only be applied on RGB image (portraits here are grayscale).
Flips are also not included as it is quite complicated to swap the predictions (some will
have to be mirrored like nose, others have to be swapped like left-right eye), and the
NN may not work properly from an upside-down portraits, in case of vertical flips
"""

# setup
test_images: List[ndarray] = read_keypoint_images()
visualizer: KeypointVisualizer = KeypointVisualizer(image_folder="facial_keypoints")
predictor_under_test: KeypointModel = KeypointModel()

contrast = gmt.create_metamorphic_relation(name="contrast", data=test_images)
brightness = gmt.create_metamorphic_relation(name="brightness", data=test_images)
both_cv2 = gmt.create_metamorphic_relation(name="both_cv2", data=test_images)
downscale = gmt.create_metamorphic_relation(name="downscale", data=test_images)
dropout = gmt.create_metamorphic_relation(name="dropout", data=test_images)
gamma = gmt.create_metamorphic_relation(name="gamma", data=test_images)
equalize = gmt.create_metamorphic_relation(name="equalize", data=test_images)
clahe = gmt.create_metamorphic_relation(name="clahe", data=test_images)
blur = gmt.create_metamorphic_relation(name="blur", data=test_images)


@gmt.transformation(contrast)
@gmt.randomized("alpha", gmt.RandFloat(0.6, 1.5))
def contrast_adjustments(image: ndarray, alpha: float) -> ndarray:
    return np.clip(alpha * image, 0, 255).astype(np.uint8)


@gmt.transformation(brightness)
@gmt.randomized("beta", gmt.RandInt(-30, 30))
def brightness_adjustments(image: ndarray, beta: int) -> ndarray:
    return np.clip(image + beta, 0, 255).astype(np.uint8)


@gmt.transformation(both_cv2)
@gmt.randomized("alpha", gmt.RandFloat(0.6, 1.5))
@gmt.randomized("beta", gmt.RandInt(-30, 30))
def cv2_brightness_contrast_adjustments(
        image: ndarray, alpha: float, beta: int
) -> ndarray:
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


@gmt.transformation(dropout)
@gmt.randomized("holes", gmt.RandInt(4, 6))
def album_dropout(
        image: ndarray, holes: int = 6, height: int = 6, width: int = 6
) -> ndarray:
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
def album_downscale(image: ndarray, scale: float = 0.5) -> ndarray:
    image_transform = albumentations.Downscale(p=1)
    return image_transform.apply(image, scale=scale, interpolation=0)


@gmt.transformation(gamma)
@gmt.randomized("limit", gmt.RandInt(70, 130))
def album_gamma(image: ndarray, limit: int = 101) -> ndarray:
    # some transform need a little different setup
    image_transform = albumentations.Compose(
        [albumentations.RandomGamma(gamma_limit=(limit, limit), p=1)]
    )
    return image_transform(image=image)["image"]


@gmt.transformation(equalize)
def album_equalize(image: ndarray) -> ndarray:
    image_transform = albumentations.Equalize(p=1)
    return image_transform.apply(image)


@gmt.transformation(clahe)
@gmt.randomized("clip_limit", gmt.RandFloat(3.0, 3.5))
def album_clahe(image: ndarray, clip_limit=3.0, tile_grid_size: int = 8) -> ndarray:
    image_transform = albumentations.CLAHE(
        clip_limit=(clip_limit, clip_limit),
        tile_grid_size=(tile_grid_size, tile_grid_size),
        p=1,
    )
    return image_transform.apply(image)


@gmt.transformation(blur)
@gmt.randomized("kernel_size", gmt.RandInt(5, 7))
def album_blur(image: ndarray, kernel_size: int = 3) -> ndarray:
    image_transform = albumentations.Blur(blur_limit=[kernel_size, kernel_size], p=1)
    return image_transform.apply(image)


@gmt.relation()
def error_is_small(f_x: Tensor, f_xt: Tensor) -> bool:
    """
    Determines if the resulting keypoints pairs are close or too far apart.

    This is measured using Mean-square-error loss function, which is the same loss function
    that is used to train this neural network by minimizing them.

    Parameters
    ----------
    f_x : Tensor
        The first set of keypoints
    f_xt : Tensor
        The second set of keypoints
    Returns
    -------
    True if the keypoints are not far apart, False otherwise.
    """
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(f_xt, f_x).item()
    return loss < 0.002


@gmt.system_under_test(
    visualize_input=visualizer.visualize_input,
    visualize_output=visualizer.visualize_output
)
def test_keypoint_predictor(image: ndarray) -> Tensor:
    """Predict the facial keypoints of a portrait"""
    return predictor_under_test.predict(image)
