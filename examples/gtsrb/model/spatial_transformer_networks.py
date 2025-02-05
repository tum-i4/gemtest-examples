"""
The Spatial Transformer Networks model structure defined in this module to load checkpoint.
It supports inference function for prediction as well. The implementation taken from
the https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html.
The paper for this model can be found in https://arxiv.org/abs/1506.02025 .
"""


from __future__ import annotations
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.nn.functional as fun
from numpy import ndarray
from torch import Tensor
from torchvision import transforms

from examples.gtsrb.data.config import NORMALIZATION_VARIABLES
from examples.gtsrb.model.config import MODEL_INPUT_SIZE
from examples.gtsrb.model.config import device

base_path = os.path.dirname(os.path.abspath(__file__))


class SpatialTransformerNetworks(nn.Module):
    """
    Init function of the Spatial Transformer Network.
    Parameters
    ----------

    Returns
    -------

    """
    model_relative_path = "bin/gtsrb_training/spatial_transformer_networks"
    model_category = 'supervised'
    model_checkpoint_name = "best_model.tar"
    model_name = "spatial_transformer_networks"

    def __init__(self):
        super(SpatialTransformerNetworks, self).__init__()
        self.num_classes = 43
        self.conv1 = nn.Conv2d(3, 100, kernel_size=5)

        self.bn1 = nn.BatchNorm2d(100)

        self.conv2 = nn.Conv2d(100, 150, kernel_size=3)

        self.bn2 = nn.BatchNorm2d(150)

        self.conv3 = nn.Conv2d(150, 250, kernel_size=3)

        self.bn3 = nn.BatchNorm2d(250)

        self.conv_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(250 * 2 * 2, 350)

        self.fc2 = nn.Linear(350, self.num_classes)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation for second linear
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def stn(self, x: Tensor) -> Tensor:
        """Spatial transformer network forward function"""
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = fun.affine_grid(theta, list(x.size()), align_corners=True)
        x = fun.grid_sample(x, grid, align_corners=True)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Process the image with the neural network."""

        x = self.stn(x)
        # Perform forward pass
        x = self.bn1(fun.max_pool2d(fun.leaky_relu(self.conv1(x)), 2))
        x = self.conv_drop(x)
        x = self.bn2(fun.max_pool2d(fun.leaky_relu(self.conv2(x)), 2))
        x = self.conv_drop(x)
        x = self.bn3(fun.max_pool2d(fun.leaky_relu(self.conv3(x)), 2))
        x = self.conv_drop(x)
        x = x.view(-1, 250 * 2 * 2)
        x = fun.relu(self.fc1(x))
        x = fun.dropout(x, training=False)
        x = self.fc2(x)
        return x

    @property
    def visualizer_name(self):
        return f"{self.model_name}_{self.model_category}_{self.model_checkpoint_name.split(':')[0]}"

    @classmethod
    def load_model(cls, model_category) -> SpatialTransformerNetworks:
        """
        load model and return instance of model
        Parameters
        ----------
        model_category : str
            which training model to load, default to 'supervised'

        Returns
        -------
        model:
            SpatialTransformerNetworks instance

        """

        cls.model_category = model_category
        models_path = os.path.join(base_path, cls.model_relative_path)
        model_path = os.path.join(models_path, model_category)
        model_file = os.path.join(model_path, "best_model.tar")

        # init STN
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
