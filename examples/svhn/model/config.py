import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input size for different model architectures
MODEL_INPUT_SIZE = {
    "mnist_cnn": (28, 28),
    "lenet": (32, 32),
    "resnet_custom": (28, 28),
    "efficient_capsnet": (28, 28),
    "wide_resnet28_2": (32, 32)
}
