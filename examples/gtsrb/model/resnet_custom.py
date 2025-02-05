"""
The Resnet Custom model structure defined in this module to load checkpoint.
It supports inference function for prediction as well. The paper of resnet can be found
in this link https://arxiv.org/abs/1512.03385 . The implementation is taken from the
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""

import os
from typing import List, Type, Union, Optional, Callable

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


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Init function of the ResNet Custom model.
    Parameters
    ----------
    block:Type[Union[BasicBlock, Bottleneck]]
        ResNet blocks
    layers:List[int]
        how many resnet block in each layer
    gray_scale
        flag whether img gray scale or rgb
    num_classes:int
        class size
    zero_init_residual:bool
        whether to initialize the residual layer
    width_per_group:int
        width of the convolution layers, we keep it as default
    replace_stride_with_dilation:bool
        flag for the stride or dilation
    norm_layer:Callable[[Tensor], nn.Module]
        The normalization layer added

    Returns
    -------

    """
    model_relative_path = "bin/gtsrb_training/resnet_custom"
    model_category = 'supervised'  # supervised, with_flip, no_flip, flip2to5_rotate90, thickening_thinning_fracture
    model_checkpoint_name = "best_model.tar"
    model_name = "resnet_custom"

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            gray_scale=False,
            num_classes: int = 43,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.layers = layers
        input_dim = 3
        if gray_scale:
            input_dim = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(input_dim, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        linear_size = 64
        if len(layers) >= 2:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
            linear_size = 128
        if len(layers) >= 3:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
            linear_size = 256
        if len(layers) >= 4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
            linear_size = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(linear_size * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if len(self.layers) >= 2:
            x = self.layer2(x)
        if len(self.layers) >= 3:
            x = self.layer3(x)
        if len(self.layers) >= 4:
            x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    @property
    def visualizer_name(self):
        return f"{self.model_name}_{self.model_category}_{self.model_checkpoint_name.split('.')[0]}"

    @classmethod
    def load_model(cls, model_category) -> "ResNet":
        """
        load model and return instance of model
        Parameters
        ----------
        model_category : str
            which training model to load, default to 'supervised'

        Returns
        -------
        model:
            ResnetCustom instance

        """

        cls.model_category = model_category
        models_path = os.path.join(base_path, cls.model_relative_path)
        model_path = os.path.join(models_path, model_category)
        model_file = os.path.join(model_path, "best_model.tar")

        # init Resnet
        block = BasicBlock
        layers = [2, 2]  # how many layers and how many block of resnet in each layers
        model = cls(block=block, layers=layers, gray_scale=False)

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
