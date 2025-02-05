from typing import Any

import cv2  # type: ignore
import matplotlib.pyplot as plt
from torchvision import transforms

from gemtest import Visualizer


class KeypointVisualizer(Visualizer):
    """
    Visualizer class for the keypoint prediction.

    """

    def __init__(self, image_folder: str) -> None:
        super().__init__()
        self.image_folder = image_folder
        self.transform = transforms.ToTensor()

    def visualize_input(self, sut_input: Any, **kwargs) -> str:
        mtc = kwargs["mtc"]
        index = kwargs["index"]
        run_id = kwargs["run_id"]
        position = kwargs["position"]

        name = f"{mtc.report.sut_name}." \
               f"{mtc.report.mr_name}." \
               f"{mtc.report.mtc_name}." \
               f"{position}_{index}.png"

        image = (self.transform(sut_input).clone() * 255).view(96, 96)
        plt.clf()
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        return self.savefig(self.image_folder, name, run_id)

    def visualize_output(self, sut_output: Any, **kwargs) -> str:
        mtc = kwargs["mtc"]
        index = kwargs["index"]
        run_id = kwargs["run_id"]
        sut_input = kwargs["sut_input"]
        position = kwargs["position"]

        name = f"{mtc.report.sut_name}." \
               f"{mtc.report.mr_name}." \
               f"{mtc.report.mtc_name}." \
               f"{position}_{index}.png"

        image = (self.transform(sut_input).clone() * 255).view(96, 96)
        plt.clf()
        plt.imshow(image, cmap="gray")
        keypoints = sut_output.clone() * 48 + 48
        plt.scatter(keypoints[:, 0], keypoints[:, 1], s=200, marker=".", c="m")
        plt.axis("off")
        return self.savefig(self.image_folder, name, run_id)
