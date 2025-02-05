import cv2
import numpy as np

from examples.gtsrb.utils.domain.image.binary_mask import BinaryMask
from examples.gtsrb.utils.processing_steps.processing_step import ImageProcessingStep


class LargestComponentFromMask(ImageProcessingStep):
    """Discards all N-1 smaller components as chromatic aberrations"""

    def apply(self, fn_input: BinaryMask) -> BinaryMask:
        total_labels, label_ids, values, centroid = cv2.connectedComponentsWithStats(
            fn_input.value
        )
        max_area = float('-inf')
        best_idx = -1
        # Label 0 is for 'background'
        for i in range(1, total_labels):
            area = values[i, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                best_idx = i
        return BinaryMask(
            (label_ids == best_idx).astype(np.uint8) *
            BinaryMask.SELECTED_PIXEL_VALUE
        )
