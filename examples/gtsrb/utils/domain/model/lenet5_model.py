import PIL.Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fun
import torchvision.transforms as transforms

from examples.gtsrb.utils.domain.model import torch_model_adapter


class _LeNet5SelfTrained(nn.Module):
    """Model trained by team-4.

    Original model has been truncated to suit the needs of team-1.
    [good_seed_detection] method has been removed + the model is no longer
    inheriting from [models.default_model.ClassificationModel] class.

    Please refer to team-4 work for the original version.
    """

    @staticmethod
    def _z_score_normalization(tensor_input, r_max=1, r_min=0):
        """
        scale the input to be between 0 and 1 (normalization)
        :param tensor_input: the layer's output tensor
        :param r_max: the upper bound of scale
        :param r_min: the lower bound of scale
        :return: scaled input
        """
        divider = tensor_input.max() - tensor_input.min()
        if divider == 0:
            return torch.zeros(tensor_input.shape)
        x_std = (tensor_input - tensor_input.min()) / divider
        x_scaled = x_std * (r_max - r_min) + r_min
        return x_scaled

    _PREPROCESS_PIPELINE = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float)
    ])

    def __init__(
            self, model_path: str, num_classes=43, input_channels=3
    ):
        super().__init__()
        # Define path for loading model weights
        self.model_name = 'lenet5_self_trained'
        self.model_path = model_path
        # Define model architecture
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        if 1 == num_classes:
            # compatible with nn.BCELoss
            self.softmax = nn.Sigmoid()
        else:
            # compatible with nn.CrossEntropyLoss
            self.softmax = nn.LogSoftmax(dim=1)

        # Load pretrained weights
        self.load_state_dict(
            torch.load(
                self.model_path,
                map_location=torch.device("cpu"),
            )
        )
        self.eval()

        self._latent_tensor_representation = None

    def forward(self, x: "PIL.Image"):
        """Preprocess input and forward pass it through network."""
        x = _LeNet5SelfTrained._PREPROCESS_PIPELINE(x)
        x = _LeNet5SelfTrained._z_score_normalization(x).unsqueeze(0)
        out = fun.relu(self.conv1(x))
        out = fun.max_pool2d(out, 2)
        out = fun.relu(self.conv2(out))
        out = fun.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = fun.relu(self.fc1(out))
        out = fun.relu(self.fc2(out))
        out = self.fc3(out)
        # Save the latent representation before softmax
        self._latent_tensor_representation = out
        out = self.softmax(out)
        return out

    @property
    def latent_vector_representation(self) -> np.ndarray:
        if self._latent_tensor_representation is None:
            raise ValueError("At least one forward pass is required")
        return torch.squeeze(self._latent_tensor_representation, 0).detach().numpy()


class Lenet5PytorchModelAdapter(torch_model_adapter.TorchModelAdapter):

    def __init__(self, model_path: str):
        super().__init__(_LeNet5SelfTrained(model_path=model_path))

    def get_latent_representation(self, one_input: "PIL.Image") -> np.ndarray:
        model_obj_cast: _LeNet5SelfTrained = self.model
        # Convert to format native to pytorch
        # Discard the output
        _ = model_obj_cast.forward(one_input)
        return model_obj_cast.latent_vector_representation
