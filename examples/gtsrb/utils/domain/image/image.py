from abc import ABC, abstractmethod
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


# TODO: Deprecated class. See rgba_image module todo for more details. Its existence
#  is justified by the existence of the rgba_image.RGBAImage class, so once the Overlay
#  operation is ported to using gtsrb_dataset approach for data loading it can be removed


class Image(ABC):
    """Abstract class representing an image with methods to read and process it."""

    def __init__(self, value: Union[str, np.ndarray]):
        if isinstance(value, str):
            self._read_from_disk(value)
        elif isinstance(value, np.ndarray):
            self._img = value
        else:
            # Copy constructor
            raise ValueError(
                'Illegal type passed to constructor: %s' % type(value)
            )

    @abstractmethod
    def _read_from_disk(self, value: str):
        raise NotImplementedError

    @property
    def value(self) -> np.ndarray:
        return self._img

    @value.setter
    def value(self, other: np.ndarray):
        assert isinstance(other, np.ndarray), 'Setter must work with Numpy array'
        self._img = other

    def plot(self):
        plt.imshow(self.value, interpolation='nearest')
        plt.show()

    def plot_in_window(self):
        """Plots the value in a separate window """
        info = f'{type(self).__name__} {self._img.shape[0]}x{self._img.shape[1]} px'
        cv2.startWindowThread()
        cv2.namedWindow(info)
        # change colors back to cv2 format
        cv2.imshow(info, cv2.cvtColor(self.value, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey()
        if key == 27 or key == 0:  # Esc or Enter
            cv2.destroyAllWindows()

    def copy(self) -> 'Image':
        return type(self)(self._img.copy())
