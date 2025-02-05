import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input size for different model architectures
MODEL_INPUT_SIZE = {
    "mnist_cnn": (32, 32),
    "lenet": (32, 32),
    "resnet_custom": (28, 28),
    "wide_resnet28_2": (32, 32),
    "spatial_transformer_networks": (32, 32),
}
