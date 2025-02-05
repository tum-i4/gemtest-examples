import os.path

import numpy as np

from examples.mnist.data.config import base_directory
from examples.mnist.utils import Dataset


def main():
    dataset = Dataset('MNIST', config_path='../config.json')

    output_relative = 'MNIST_Complete/MNIST_Images/MNIST/Final_Test/Images/'
    output_absolute = os.path.join(base_directory, output_relative)
    os.makedirs(output_absolute, exist_ok=True)

    counter = 0
    for image in dataset.X_test:
        output_path = output_absolute + str(counter) + '.ppm'
        save_grayscale_ppm(image, output_path)
        counter = counter + 1


def save_grayscale_ppm(image_array, file_path):
    """
    Save a (28, 28, 1) grayscale NumPy ndarray as a PPM file.

    Parameters:
    - image_array: NumPy ndarray representing the grayscale image.
    - file_path: String, the path to save the PPM file.
    """
    if np.max(image_array) <= 1.0:
        # Ensure the pixel values are in the range [0, 255]. *256 indstead of 255 (have a look at pre_process_mnist.py pre_process function)
        image_array = (image_array * 256).astype(np.uint8)

    height, width, channels = image_array.shape

    if channels != 1:
        raise ValueError("The input array must have shape (height, width, 1) for grayscale images.")

    max_val = np.amax(image_array)
    header = f"P2\n{width} {height}\n{max_val}\n"
    data = image_array[:, :, 0].flatten()

    with open(file_path, "w") as ppm_file:
        ppm_file.write(header)
        np.savetxt(ppm_file, data, fmt="%d", delimiter=" ")


if __name__ == "__main__":
    main()
