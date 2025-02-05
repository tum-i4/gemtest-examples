# Dict specifying used image sizes for model training
IMG_SIZE = {
    "gtsrb": 32,
}

# Mean and standard deviation values used for normalizing input images
NORMALIZATION_VARIABLES = {
    "gtsrb": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    }
}
