import cv2
import numpy as np
from matplotlib import pyplot as plt


class BinaryMask:
    """Binary mask used for selecting areas of an RGB image."""

    SELECTED_PIXEL_VALUE = 255  # WHITE
    UNSELECTED_PIXEL_VALUE = 0  # UNSELECTED

    def __init__(self, value: np.ndarray):
        self._mask = value

    @property
    def value(self):
        return self._mask

    @value.setter
    def value(self, other):
        assert isinstance(other, BinaryMask)
        self._mask = other.value

    def union(self, another: 'BinaryMask') -> None:
        """Calculate the union of two masks. Modification is done in-place."""
        assert self.value.shape == another.value.shape, \
            f"Dimension mismatch: {self.value.shape} {another.value.shape}"
        self._mask = cv2.bitwise_or(self._mask, another.value)

    def xor(self, another: 'BinaryMask') -> 'BinaryMask':
        """Returns part of binary mask that excludes the 'another'."""
        assert self.value.shape == another.value.shape, \
            f"Dimension mismatch: {self.value.shape} {another.value.shape}"
        return BinaryMask(cv2.bitwise_xor(self._mask, another.value))

    def negate(self) -> 'BinaryMask':
        """Returns the opposite binary mask."""
        return BinaryMask(cv2.bitwise_not(self._mask))

    def erode(self, kernel_size: int = 3) -> 'BinaryMask':
        """Erodes the binary mask."""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return BinaryMask(cv2.erode(self._mask, kernel, iterations=1))

    def copy(self) -> 'BinaryMask':
        return BinaryMask(self._mask.copy())

    def plot(self):
        plt.imshow(self.value, interpolation='nearest')
        plt.show()
