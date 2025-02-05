import cv2
import numpy as np

from examples.gtsrb.utils.domain.image.image import Image


# TODO: This class is deprecated and kept for compatibility with overlay operation.
#  The orthodox approach for loading data is through the gtsrb_dataset.GTSRBDataset
#  class with 'RGBA' passed as value for the 'image_mode' constructor argument

class RGBAImage(Image):
    """An RGBA image which is used for overlaying on top of a traffic sign"""

    def _read_from_disk(self, value: str):
        # IMREAD_UNCHANGED opens the image with the alpha channel
        self._img: np.ndarray = cv2.imread(value, cv2.IMREAD_UNCHANGED)
        self._img = cv2.cvtColor(self._img, cv2.COLOR_BGRA2RGBA)

    def show(self):
        super().show(cv2.COLOR_RGBA2BGRA)
