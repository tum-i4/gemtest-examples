import os.path
from pathlib import Path

from examples.mnist.data.config import *


def get_test_image_paths():
    """
    Walk through folder that includes all images and return full paths of test images
    """
    complete_path = (base_directory / Path("MNIST_Complete", "MNIST_Images",
                                           "MNIST", "Final_Test", "Images"))
    retraining_path = (base_directory / Path("MNIST_Sample", "MNIST_Images",
                                             "MNIST", "Images_Label_Preserving"))
    sample_path = (base_directory / Path("MNIST_Sample", "MNIST_Images",
                                         "MNIST", "Final_Test", "Images"))
    if complete_path.is_dir():
        data_folder_path = complete_path
    elif retraining_path.is_dir():
        data_folder_path = retraining_path
        print("using the fixmatch retraining dataset which is chosen specifically from the test dataset.")
    else:
        data_folder_path = sample_path
        print("using the sample dataset.")

    file_paths = []
    for root, dirs, files in os.walk(data_folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths
