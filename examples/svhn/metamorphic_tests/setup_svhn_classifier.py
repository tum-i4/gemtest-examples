"""
This example demonstrates MR tests of a traffic sign classifier NN:
- if the images are perturbed slightly, the prediction should not differ by much.
- if the perturbation causes the semantic information to change,
  e.g. from left to right turn, the prediction should reflect this change.

What follows here are 16 image perturbation functions, all with the same signature: receives
an image, and one or more parameters that can be used on the perturbation function.
Consult albumentations documentation for more information on functions that uses them:
https://albumentations.ai/docs/api_reference/augmentations/transforms/.
Note that the randomization of some of the parameters are delegated into our own framework
instead of using the perturbation's own random functionality.
For demonstration purposes, two MR tests will make use of two or three of these
perturbations in random sequence.
"""

import datetime
import json
import logging
import uuid
from random import randint
from typing import List, Any

import albumentations  # type: ignore
import cv2  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from PIL import Image
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

import gemtest as gmt
from examples.svhn.data.config import *
from examples.svhn.data.utils import get_test_image_paths, load_image_resource
from examples.svhn.model.wide_resnet28_2 import WideResNet
from examples.svhn.model.mnist_cnn import MnistCNN
from examples.svhn.model.lenet import LeNet
from examples.svhn.model.resnet_custom import ResNet


# Version before Practicum in WS 2023/24
VERSION = 2.0
base_directory = os.path.dirname(os.path.abspath(__file__))


class ExceptionLogger:
    """Class to help log exceptions that occur when saving images for visualization."""

    def __init__(self):
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())

    def logexception_geterrorstring(self, e: Exception) -> str:
        self.logger.error(e)
        return f"Failed to save image: {str(e)}"


class SVHNVisualizer(gmt.Visualizer):
    labels_dict = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
    }

    def __init__(self, image_folder: str) -> None:
        super().__init__()
        self.image_folder = image_folder

    def denormalize(self, tensor, mean, std):
        """ Reverses the normalization on a tensor. """
        # Cloning the tensor is important to not modify the original tensor
        tensor = tensor.clone()

        # Denormalize each channel
        # for t, m, s in zip(tensor, mean, std):
        tensor.mul_(std).add_(mean)
        return tensor

    def normalized_tensor_to_numpy(self, normalized_tensor) -> np.ndarray:
        """
        denormalize tensor and return it as a numpy array
        """

        # Denormalize
        denormalized_tensor = self.denormalize(normalized_tensor, **NORMALIZATION_VARIABLES)

        # return numpy array
        return denormalized_tensor.detach().cpu().numpy()

    def visualize_input(self, sut_input: Any, **kwargs) -> str:
        # sut_input = self.normalized_tensor_to_numpy(sut_input)
        mtc = kwargs["mtc"]
        index = kwargs["index"]
        run_id = kwargs["run_id"]
        position = kwargs["position"]

        name = f"{mtc.report.sut_name}." \
               f"{mtc.report.mr_name}." \
               f"{mtc.report.mtc_name}." \
               f"{position}_{index}.png"

        # amir changed. Generalization. It also works for grayscale images with one channel.
        if sut_input.shape[-1] in (3, 4):  # RGB or RGBA image
            return self.imsave(sut_input,
                               image_folder=self.image_folder,
                               image_name=name,
                               run_id=run_id)
        elif sut_input.shape[-1] == 1:  # Grayscale image
            # convert it to 3 channels
            # before saving, or save it as a single-channel image, based on your preference.
            # Here, I'll convert it to 3 channels using cv2.cvtColor
            sut_input_uint8 = sut_input.astype('uint8')
            grayscale_image = cv2.cvtColor(sut_input_uint8, cv2.COLOR_GRAY2RGB)
            out = self.imsave(grayscale_image, image_folder=self.image_folder, image_name=name, run_id=run_id)
            return out
        else:
            raise ValueError("Unsupported number of channels in the input image")

    def visualize_output(self, sut_output: Any, **kwargs) -> str:
        return self.labels_dict.get(sut_output, f"unknown: {sut_output}")


def export_data(report: gmt.GeneralMTCExecutionReport):
    data_path = os.path.join(base_directory, "../assets/data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if report is None:
        return exception_logger.logexception_geterrorstring(TypeError("MTCExecutionReport is None"))

    # currently only uses the first source / followup of a MR -
    source_input = report.source_inputs[0]
    followup_input = report.followup_inputs[0]
    source_output = report.source_outputs[0]
    followup_output = report.followup_outputs[0]

    if isinstance(followup_input, str) and followup_input == "(not generated)":
        return exception_logger.logexception_geterrorstring(ValueError("No data to export"))

    if not isinstance(followup_input, np.ndarray):
        raise TypeError(f"Svhn Data exporter requires the "
                        f"followup-input to be of type np.ndarray. The provided object "
                        f"is of type {type(followup_input)} with value \"{followup_input}\".")

    image_name = report.transformation_name
    for key, value in report.parameters.items():
        image_name += f"_{key}_{value}"
    image_name += f"{uuid.uuid4()}.png"

    path = data_path + "/" + image_name
    json_path = path.replace(".png", ".json")

    assert source_input.shape == followup_input.shape, \
        f"Image dimensions are not correct: Got source-dim={source_input.shape} " \
        f"and follow-up-dim={followup_input.shape} for metamorphic relation " \
        f"{report.transformation_name}."

    ssim_score = ssim(
        np.array(source_input).astype(float) / 255,
        np.array(followup_input).astype(float) / 255,
        data_range=1 - 0,
        channel_axis=2
    )
    mse = mean_squared_error(
        np.array(source_input).astype(float) / 255,
        np.array(followup_input).astype(float) / 255
    )
    l_0_norm = np.average(
        np.linalg.norm(
            source_input.astype("float") - followup_input.astype("float"),
            ord=0,
            axis=2
        )
    )
    l_2_norm = np.average(
        np.linalg.norm(
            source_input.astype("float") - followup_input.astype("float"),
            ord=2,
            axis=2
        )
    )
    l_inf_norm = np.average(
        np.linalg.norm(
            source_input.astype("float") - followup_input.astype("float"),
            ord=np.inf,
            axis=2
        )
    )

    # if source_output != followup_output:
    # data about image prediction
    original_image_name = ""
    for i in range(len(test_image_paths)):
        original_img = load_image_resource(test_image_paths[i])
        if np.array_equal(original_img, source_input):
            original_image_name = test_image_paths[i]
            break

    rand = randint(1000, 10000)
    source_input_no_channel = source_input.reshape(IMG_SIZE["svhn"], IMG_SIZE["svhn"])
    im = Image.fromarray(source_input_no_channel)
    im.save(
        data_path + "/" + str(rand) + "_source_pred=" + str(source_output) + "_" + original_image_name.split("/")[-1])

    followup_input_no_channel = followup_input.reshape(IMG_SIZE["svhn"], IMG_SIZE["svhn"])
    im = Image.fromarray(followup_input_no_channel)
    im.save(
        data_path + "/" + str(rand) + "_followup_pred=" + str(followup_output) + "_" + original_image_name.split("/")[
            -1])

    predicted_org_probability = classifier_under_test.evaluate_image_softmax(
        source_input).flatten()[source_output].item()

    predicted_transform_probability = classifier_under_test.evaluate_image_softmax(
        followup_input).flatten()[followup_output].item()

    expected_transform_labels = []
    expected_transform_label_names = []

    for svhn_sign_label in range(len(svhn_visualizer.labels_dict.keys())):
        if source_output == svhn_sign_label:
            expected_transform_labels.append(svhn_sign_label)
            expected_transform_label_names.append(svhn_visualizer.visualize_output(svhn_sign_label))

    header = dict(__version_entry__=[
        dict(__Tool__='{}'.format(classifier_under_test.visualizer_name),
             __Version__=VERSION,
             __Time__=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ])

    gt_labels = dict(__gt_labels__=[
        dict(org_pred_class="NA",
             org_pred_classname="NA")
    ])

    transform_labels = dict(__transform_labels__=[
        dict(__class_id__=expected_transform_labels,
             __class_name__=expected_transform_label_names)
    ])

    add_info_1 = dict(__original_prediction__=[
        dict(org_pred_class=source_output,
             org_pred_classname=source_output,
             org_pred_prob=predicted_org_probability)
    ])

    add_info_2 = dict(__current_predictions__=[
        dict(op_class=followup_output,
             op_classname=followup_output,
             op_prob=predicted_transform_probability)
    ])

    add_info_3 = dict(__image_quality_metrics__=[
        dict(org_image=original_image_name,
             SSIM=round(float(ssim_score), 2),
             MSE=round(float(mse), 2),
             L0_norm=int(l_0_norm),
             L2_norm=round(float(l_2_norm), 2),
             Linf_norm=l_inf_norm,
             transformations=report.transformation_name)
    ])
    file = {**header, **gt_labels, **transform_labels, **add_info_1, **add_info_2,
            **add_info_3}
    with open(os.path.join(json_path), 'w') as json_file:
        json.dump(file, json_file, indent=4)


# "label_preserving" , "none_label_preserving"
test_image_paths: List[str] = get_test_image_paths()

# model_categories: supervised, fixmatch
classifier_under_test: WideResNet = WideResNet.load_model(model_category="fixmatch")
# classifier_under_test: MnistCNN = MnistCNN.load_model(model_category="fixmatch")
# classifier_under_test: ResNet = ResNet.load_model(model_category="fixmatch")
# classifier_under_test: LeNet = LeNet.load_model(model_category="fixmatch")

exception_logger: ExceptionLogger = ExceptionLogger()
svhn_visualizer = SVHNVisualizer(classifier_under_test.visualizer_name)
