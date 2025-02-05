import os

# Dict specifying used image sizes for model training
IMG_SIZE = {
    "svhn": 28,
}

# Mean and standard deviation values used for normalizing input images
NORMALIZATION_VARIABLES = {
    "svhn": {
        "mean": (0.4382, 0.4450, 0.4753),
        "std": (0.1166, 0.1192, 0.1017,)}
}

base_directory = os.path.dirname(os.path.abspath(__file__))
