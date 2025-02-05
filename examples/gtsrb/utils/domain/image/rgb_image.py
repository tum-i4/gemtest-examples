import PIL.Image
import cv2
import numpy as np
from matplotlib import pyplot as plt

from examples.gtsrb.utils.domain.image.binary_mask import BinaryMask


class RGBImage:
    """Wrapper around images outputted by GTSRBDataset class.

    Assumes np.ndarray received in constructor is in RGB format.
    """

    def __init__(
            self,
            value: np.ndarray
    ):
        self._img = value

    def to_grayscale(self) -> 'RGBImage':
        grayscale_np: np.ndarray = cv2.cvtColor(self.value, cv2.COLOR_RGB2GRAY)
        return RGBImage(grayscale_np)

    @staticmethod
    def read_from_disk(path: str) -> "RGBImage":
        # PyCharm panics about np.array conversion
        # noinspection PyTypeChecker
        value = np.array(PIL.Image.open(path).convert("RGB"))
        return RGBImage(value)

    @property
    def value(self):
        return self._img

    @value.setter
    def value(self, other):
        # assert isinstance(other, RGBImage)
        # self._img = other.value
        assert isinstance(other, np.ndarray)
        self._img = other

    def plot(self):
        """Displays RGB image."""
        plt.imshow(self._img, interpolation='nearest')
        plt.show()

    def copy(self) -> 'RGBImage':
        return RGBImage(self._img.copy())

    def get_crop(self, mask: BinaryMask) -> 'RGBImage':
        """Gets new RGB image where only pixels selected by mask are included."""
        assert self.value.shape[0:2] == mask.value.shape, (
            f"Dimension mismatch: RGB {self.value.shape[0:2]} vs BinaryMask {mask.value.shape}"
        )
        return RGBImage(np.array(
            cv2.bitwise_and(self._img, self._img, mask=mask.value)
        ))

    def overlay(self, on_top: 'RGBImage') -> 'RGBImage':
        """Overlays two RGB images. Background is where black pixels are (0,0,0)."""
        assert self.value.shape == on_top.value.shape, (
            f"Dimension mismatch: Background {self.value.shape} "
            f"vs Foreground {on_top.value.shape}"
        )
        # if self has pixel value self.value[:,:,] == (0,0,0) then we insert the corresponding
        # pixel value of on_top.value. Create a mask for the (0, 0, 0) pixels in the background
        # image
        mask = (self.value == [0, 0, 0]).all(axis=2)
        self.value[mask] = on_top.value[mask]
        return RGBImage(self.value)
