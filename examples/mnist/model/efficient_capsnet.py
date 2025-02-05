"""
The Efficient capsnet model structure defined in this module to load checkpoint.
It supports inference function for prediction as well. We get the implementation from here
 https://github.com/adambielski/CapsNet-pytorch/blob/master/net.py. The original paper can be found here:
https://arxiv.org/abs/2101.12491
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
from torch.autograd import Variable
from torchvision import transforms

from examples.mnist.data.config import NORMALIZATION_VARIABLES
from examples.mnist.model.config import MODEL_INPUT_SIZE
from examples.mnist.model.config import device

base_path = os.path.dirname(os.path.abspath(__file__))


def squash(input, eps=10e-21):
    n = torch.norm(input, dim=-1, keepdim=True)
    return (1 - 1 / (torch.exp(n) + eps)) * (input / (n + eps))


def length(input):
    return torch.sqrt(torch.sum(input ** 2, dim=-1) + 1e-8)


def mask(input):
    if type(input) is list:
        input, mask = input
    else:
        x = torch.sqrt(torch.sum(input ** 2, dim=-1))
        mask = F.one_hot(torch.argmax(x, dim=1), num_classes=x.shape[1]).float()

    masked = input * mask.unsqueeze(-1)
    return masked.view(input.shape[0], -1)


class PrimaryCapsLayer(nn.Module):
    """Create a primary capsule layer where the properties of each capsule are extracted
    using a 2D depthwise convolution.

    Args:
        in_channels (int): depthwise convolution's number of features
        kernel_size (int): depthwise convolution's kernel dimension
        num_capsules (int): number of primary capsules
        dim_capsules (int): primary capsule dimension
        stride (int, optional): depthwise convolution's strides. Defaults to 1.
    """

    def __init__(self, in_channels, kernel_size, num_capsules, dim_capsules, stride=1):
        super(PrimaryCapsLayer, self).__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            padding="valid",
        )
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules

    def forward(self, input):
        output = self.depthwise_conv(input)
        output = output.view(output.size(0), self.num_capsules, self.dim_capsules)
        return squash(output)


class RoutingLayer(nn.Module):
    """Self-attention routing layer using a fully-connected network, to create a parent
    layer of capsules.

    Args:
        num_capsules (int): number of primary capsules
        dim_capsules (int): primary capsule dimension
    """

    def __init__(self, num_capsules, dim_capsules):
        super(RoutingLayer, self).__init__()
        self.W = nn.Parameter(torch.Tensor(num_capsules, 16, 8, dim_capsules))
        self.b = nn.Parameter(torch.zeros(num_capsules, 16, 1))
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, input):
        u = torch.einsum(
            "...ji,kjiz->...kjz", input, self.W
        )  # u shape = (None, num_capsules, height*width*16, dim_capsules)
        c = torch.einsum("...ij,...kj->...i", u, u)[
            ..., None
        ]  # b shape = (None, num_capsules, height*width*16, 1) -> (None, j, i, 1)
        c = c / torch.sqrt(
            torch.Tensor([self.dim_capsules]).type(torch.cuda.FloatTensor)
        )
        c = torch.softmax(c, axis=1)
        c = c + self.b
        s = torch.sum(
            torch.mul(u, c), dim=-2
        )  # s shape = (None, num_capsules, dim_capsules)
        return squash(s)


class EfficientCapsNet(nn.Module):
    """Efficient-CapsNet architecture implementation.

    Args:
        nn :Torch.nn.Module
    """
    input_size = (28, 28)

    def __init__(self):
        super(EfficientCapsNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=5, padding="valid"
        )
        self.batch_norm1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding="valid")
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding="valid")
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding="valid")
        self.batch_norm4 = nn.BatchNorm2d(128)

        self.primary_caps = PrimaryCapsLayer(
            in_channels=128, kernel_size=9, num_capsules=16, dim_capsules=8
        )
        self.digit_caps = RoutingLayer(num_capsules=10, dim_capsules=16)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)

    def forward(self, x):
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = torch.relu(self.batch_norm3(self.conv3(x)))
        x = torch.relu(self.batch_norm4(self.conv4(x)))
        x = self.primary_caps(x)
        x = self.digit_caps(x)
        probs = length(x)
        return x, probs


class ReconstructionNet(nn.Module):
    """
    Reconstruction network for after the efficient capsnet. 2 Linear layers are added.
    This network reconstruct the input (expl: the mnist digit , check the fc3 output 28x28=784 )
    """

    def __init__(self, n_dim=16, n_classes=10):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim * n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        x = mask(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.view(-1, 784)


class MarginLoss(nn.Module):
    """
    Margin loss is the special loss for the efficient capsnet
    """

    def __init__(self, m_pos=0.9, m_neg=0.1, lambda_=0.5):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, y_pred, y_true, reduction=None):
        # y_pred shape is [16,10], while y_true is [16]
        t = torch.zeros(y_pred.size()).long()
        if y_true.is_cuda:
            t = t.cuda()
        t = t.scatter_(1, y_true.data.view(-1, 1), 1)
        targets = Variable(t)
        losses = targets * torch.pow(
            torch.clamp(self.m_pos - y_pred, min=0.0), 2
        ) + self.lambda_ * (1 - targets) * torch.pow(
            torch.clamp(y_pred - self.m_neg, min=0.0), 2
        )
        losses = torch.sum(losses, dim=1)
        if reduction == "mean":
            return losses.mean()
        if reduction == "sum":
            return losses.sum()

        return losses


class EfficientCapsNetWithReconstruction(nn.Module):
    """
    Full network, efficient capsnet + reconstruction net.
    """
    input_size = (28, 28)

    model_relative_path = "bin/mnist_training/efficient_capsnet"
    model_category = 'supervised'  # fixmatch
    model_checkpoint_name = "best_model.tar"
    model_name = "efficient_capsnet"

    def __init__(self, efficient_capsnet, reconstruction_net):
        super(EfficientCapsNetWithReconstruction, self).__init__()
        self.efficient_capsnet = efficient_capsnet
        self.reconstruction_net = reconstruction_net

    def forward(self, x):
        x, probs = self.efficient_capsnet(x)
        # reconstruction = self.reconstruction_net(x) # dont need to calculate
        return probs

    @property
    def visualizer_name(self):
        return f"{self.model_name}_{self.model_category}_{self.model_checkpoint_name.split(':')[0]}"

    @classmethod
    def load_model(cls, model_category) -> EfficientCapsNetWithReconstruction:
        """load model and return instance of model"""

        cls.model_category = model_category
        models_path = os.path.join(base_path, cls.model_relative_path)
        model_path = os.path.join(models_path, model_category)
        model_file = os.path.join(model_path, "best_model.tar")

        # init Efficient Capsnet with Construction Layer
        efficient_capsnet = EfficientCapsNet()
        reconstraction_net = ReconstructionNet()

        model = cls(efficient_capsnet, reconstraction_net)

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
            transforms.Normalize(**NORMALIZATION_VARIABLES["mnist"]),
            transforms.Resize(MODEL_INPUT_SIZE[self.model_name])
        ])(img).unsqueeze(0).to(device)

    def evaluate_image(self, img: ndarray) -> int:
        """Process the image with the neural network, and return the most likely class."""

        tensor_img: Tensor = self.preprocess(img)
        logits: Tensor = self.forward(tensor_img)
        return logits.max(dim=1)[1].item()

    def evaluate_image_softmax(self, img: ndarray) -> Tensor:
        """Process the image with the neural network, and return the softmax values."""
        tensor_img: Tensor = self.preprocess(img)
        logits: Tensor = self.forward(tensor_img)
        return fun.softmax(logits, dim=1).detach()
