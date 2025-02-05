import os.path
from pathlib import Path

import cv2

from examples.svhn.data.config import *
from gemtest.metamorphic_error import InvalidInputError


def get_test_image_paths():
    """
    Walk through folder that includes all images and return full paths of test images
    """
    if (base_directory / Path("SVHN_Complete")).is_dir():
        data_folder_path = base_directory / Path("SVHN_Complete", "SVHN_Images",
                                                 "SVHN", "Final_Test")
    else:
        data_folder_path = base_directory / Path("SVHN_Sample", "SVHN_Images",
                                                 "SVHN", "label_preserving")

        print("using the sample dataset which is chosen specifically from the test dataset.")
    # Get the list of files in the folder
    file_paths = []
    for root, dirs, files in os.walk(data_folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


def load_image_resource(input_path: str):
    """
    Loads an image resource from a file path. Raises an InvalidInputError if the file is
    not a valid image file. Returns a numpy array.
    """
    file_extension = Path(input_path).suffix.lower()
    # expand to more file formats if required
    if file_extension not in (".png", ".jpg", ".jpeg", ".ppm"):
        raise InvalidInputError(f"This data loader only supports lazy loading of .png, "
                                f".jpg, .jpeg and .ppm files. File at {input_path} is "
                                f"not supported.")

    image = cv2.imread(input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (IMG_SIZE["svhn"], IMG_SIZE["svhn"]))
    return resized_image
