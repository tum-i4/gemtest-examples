"""
The WideRsnet28-2 model structure defined in this module to load checkpoint.
It supports inference function for prediction as well. The implementation taken from
the https://github.com/fbuchert/fixmatch-pytorch. The paper for this model can be found in
https://arxiv.org/abs/1605.07146 .
"""

from __future__ import annotations
from __future__ import print_function

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as fun
from numpy import ndarray
from torch import Tensor
from torchvision import transforms

from examples.mnist.data.config import NORMALIZATION_VARIABLES
from examples.mnist.model.config import MODEL_INPUT_SIZE
from examples.mnist.model.config import device

base_path = os.path.dirname(os.path.abspath(__file__))


class BasicBlock(nn.Module):
    def __init__(
            self,
            in_planes,
            out_planes,
            stride,
            dropout_rate=0.0,
            activate_before_residual=False,
    ):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.dropout_rate = dropout_rate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
                (not self.equalInOut)
                and nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=True,
        )
                or None
        )
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(
            self,
            nb_layers,
            in_planes,
            out_planes,
            block,
            stride,
            dropout_rate=0.0,
            activate_before_residual=False,
    ):
        super(NetworkBlock, self).__init__()
        self.layer = NetworkBlock._make_layer(
            block,
            in_planes,
            out_planes,
            nb_layers,
            stride,
            dropout_rate,
            activate_before_residual,
        )

    @staticmethod
    def _make_layer(
            block,
            in_planes,
            out_planes,
            nb_layers,
            stride,
            dropout_rate,
            activate_before_residual,
    ):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropout_rate,
                    activate_before_residual,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """
   Init function of the ResNet Custom model.
   Parameters
   ----------
   depth:Type[Union[BasicBlock, Bottleneck]]
       how many BasicBlock or Bottleneck
   widen_factor:List[int]
       widen the conv. layer
   dropout_rate:float
       apply dropout if bigger than 0
   num_classes:int
       class size

   Returns
   -------

   """

    model_relative_path = "bin/mnist_training/wide_resnet28_2"
    model_category = 'supervised'  # supervised, with_flip, no_flip, flip2to5_rotate90, thickening_thinning_fracture
    model_checkpoint_name = "best_model.tar"
    model_name = "wide_resnet28_2"

    def __init__(self, depth, num_classes, widen_factor=1, dropout_rate=0.0):
        super(WideResNet, self).__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            1, n_channels[0], kernel_size=3, stride=1, padding=1, bias=True
        )
        # 1st block
        self.block1 = NetworkBlock(
            n,
            n_channels[0],
            n_channels[1],
            block,
            1,
            dropout_rate,
            activate_before_residual=True,
        )
        # 2nd block
        self.block2 = NetworkBlock(
            n, n_channels[1], n_channels[2], block, 2, dropout_rate
        )
        # 3rd block
        self.block3 = NetworkBlock(
            n, n_channels[2], n_channels[3], block, 2, dropout_rate
        )
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(n_channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(n_channels[3], num_classes)
        self.nChannels = n_channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)

    def get_embedding_dim(self):
        return self.nChannels

    @property
    def visualizer_name(self):
        return f"{self.model_name}_{self.model_category}_{self.model_checkpoint_name.split(':')[0]}"

    @classmethod
    def load_model(cls, model_category) -> WideResNet:
        """
        load model and return instance of model
        Parameters
        ----------
        model_category : str
            which training model to load, default to 'supervised'
        Returns
        -------
        model:
            WideResNet instance

        """

        cls.model_category = model_category
        models_path = os.path.join(base_path, cls.model_relative_path)
        model_path = os.path.join(models_path, model_category)
        model_file = os.path.join(model_path, "best_model.tar")

        # init Wide_Resnet28_2
        model = cls(depth=28, widen_factor=2, dropout_rate=0, num_classes=10)

        # load model to device: cpu or gpu
        torch_device = torch.device(device)
        checkpoint = torch.load(model_file, map_location=torch_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(torch_device)

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
        max_probs, preds = fun.softmax(logits, dim=1).max(dim=1)
        return preds.item()

    def evaluate_image_softmax(self, img: ndarray) -> Tensor:
        """Process the image with the neural network, and return the softmax values."""
        tensor_img: Tensor = self.preprocess(img)
        logits: Tensor = self.forward(tensor_img)
        return fun.softmax(logits, dim=1).detach()
