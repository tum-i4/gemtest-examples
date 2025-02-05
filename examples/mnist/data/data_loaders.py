from pathlib import Path

import cv2  # type: ignore
import numpy as np

from gemtest.metamorphic_error import InvalidInputError


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

    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    normalized = np.max(image) <= 1.0
    if normalized:
        image = (image * 256).astype(np.int8)

    image = np.expand_dims(image, axis=-1)
    return image
