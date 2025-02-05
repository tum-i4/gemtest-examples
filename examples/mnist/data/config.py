import os

# Dict specifying used image sizes for model training
IMG_SIZE = {
    "mnist": 28,
}

# Mean and standard deviation values used for normalizing input images
NORMALIZATION_VARIABLES = {
    "mnist": {
        "mean": (0.1307,),
        "std": (0.3081,),
    }
}

base_directory = os.path.dirname(os.path.abspath(__file__))
dataset_sample_path = os.path.join(base_directory, "MNIST_Sample/")
dataset_complete_path = os.path.join(base_directory, "MNIST_Complete/")
