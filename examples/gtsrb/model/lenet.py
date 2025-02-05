"""
The Lenet model structure defined in this module to load checkpoint.
It supports inference function for prediction as well. We get the implementation architecture
from https://en.wikipedia.org/wiki/LeNet . The original paper can be found here:
http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
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

from examples.gtsrb.data.config import NORMALIZATION_VARIABLES
from examples.gtsrb.model.config import MODEL_INPUT_SIZE
from examples.gtsrb.model.config import device

base_path = os.path.dirname(os.path.abspath(__file__))


class LeNet(nn.Module):
    """
    The implementation of the LeNet model.
    """
    model_relative_path = "bin/gtsrb_training/lenet"
    model_category = 'supervised'  # supervised, with_flip, no_flip, flip2to5_rotate90, thickening_thinning_fracture
    model_checkpoint_name = "best_model.tar"
    model_name = "lenet"

    def __init__(self):
        super(LeNet, self).__init__()
        self.num_classes = 43
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(3, 6, 5, 1, 0)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        """Model structure"""
        x = self.conv1(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, (2, 2))

        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, (2, 2))

        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    @property
    def visualizer_name(self):
        return f"{self.model_name}_{self.model_category}_{self.model_checkpoint_name.split(':')[0]}"

    @classmethod
    def load_model(cls, model_category) -> LeNet:
        """
        load model and return instance of model
        Parameters
        ----------
        model_category : str
            which training model to load, default to 'supervised'
        Returns
        -------
        model:
            LeNet instance

        """

        cls.model_category = model_category
        models_path = os.path.join(base_path, cls.model_relative_path)
        model_path = os.path.join(models_path, model_category)
        model_file = os.path.join(model_path, "best_model.tar")

        # init LeNet
        model = cls()

        # load model to device: cpu or gpu
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
            transforms.Normalize(**NORMALIZATION_VARIABLES["gtsrb"]),
            transforms.Resize(MODEL_INPUT_SIZE[self.model_name])
        ])(img).unsqueeze(0).to(device)

    def evaluate_image(self, img: ndarray) -> int:
        """Process the image with the neural network, and return the most likely class."""

        tensor_img: Tensor = self.preprocess(img)
        logits: Tensor = self.forward(tensor_img)
        max_probs, preds = fun.softmax(logits, dim=1).max(dim=1)
        return preds.item()

    def evaluate_image_softmax(self, img: ndarray) -> Tensor:
        """Process the image with the neural network, and return the softmax values."""
        tensor_img: Tensor = self.preprocess(img)
        logits: Tensor = self.forward(tensor_img)
        return fun.softmax(logits, dim=1).detach()
