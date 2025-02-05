"""
MNIST CNN model structure defined in this module to load checkpoint.
It supports inference function for prediction as well.
"""

from __future__ import annotations
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as fun
from numpy import ndarray
from torch import Tensor
from torchvision import transforms

from examples.svhn.data.config import NORMALIZATION_VARIABLES
from examples.svhn.model.config import MODEL_INPUT_SIZE
from examples.svhn.model.config import device

base_path = os.path.dirname(os.path.abspath(__file__))


class MnistCNN(nn.Module):
    """
    Defines custom vanilla model for SVHN dataset. It is the same model as in MNIST
    """
    model_relative_path = "bin/svhn_training/mnist_cnn"
    model_category = 'supervised'  # supervised, with_flip, no_flip, flip2to5_rotate90, thickening_thinning_fracture
    model_checkpoint_name = "best_model.tar"
    model_name = "mnist_cnn"

    def __init__(self):
        super(MnistCNN, self).__init__()
        self.num_classes = 10
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

    @property
    def visualizer_name(self):
        return f"{self.model_name}_{self.model_category}_{self.model_checkpoint_name.split(':')[0]}"

    @classmethod
    def load_model(cls, model_category) -> MnistCNN:
        """
        load model and return instance of model
        Parameters
        ----------
        model_category : str
            which training model to load, default to 'supervised'

        Returns
        -------
        model:
            MnistCNN instance

        """

        cls.model_category = model_category
        models_path = os.path.join(base_path, cls.model_relative_path)
        model_path = os.path.join(models_path, model_category)
        model_file = os.path.join(model_path, "best_model.tar")

        # init MnistCNN
        model = cls()

        # load model to device: cpu,gpu
        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        # put model to evaluation mode for testing.(it discards the BN, Dropout)
        model.eval()

        return model

    def preprocess(self, img: ndarray) -> Tensor:
        """Normalize the input image with std and mean and resize
        according to model expected size then return as batched tensor
        to feed to the model.
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**NORMALIZATION_VARIABLES["svhn"]),
            transforms.Resize(MODEL_INPUT_SIZE[self.model_name])
        ])(img).unsqueeze(0).to(device)

    def evaluate_image(self, img: ndarray) -> int:
        """Process the image with the neural network, and return the most likely class."""
        tensor_img: Tensor = self.preprocess(img)
        logits: Tensor = self.forward(tensor_img)
        return fun.softmax(logits, dim=1).max(dim=1)[1].item()

    def evaluate_image_softmax(self, img: ndarray) -> Tensor:
        """Process the image with the neural network, and return the softmax values."""
        tensor_img: Tensor = self.preprocess(img)
        logits: Tensor = self.forward(tensor_img)
        return fun.softmax(logits, dim=1).detach()
