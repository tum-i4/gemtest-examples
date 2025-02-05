import numpy as np

from examples.gtsrb.utils.domain.image.binary_mask import BinaryMask
from examples.gtsrb.utils.processing_steps.processing_step import ImageProcessingStep


class SelectMaskEdgeOnly(ImageProcessingStep):
    """Get edge points of a binary mask.

    An edge point is defined as a point where the gradient of
    a value is different from 0 i.e. the selected white pixels (value =255)
    neighbour black unselected pixels (value =0), resulting in non-zero gradient.
    """

    def apply(self, fn_input: BinaryMask) -> BinaryMask:
        gx, gy = np.gradient(fn_input.value)
        temp_edge = gy ** 2 + gx ** 2
        temp_edge[temp_edge != 0] = 255
        return BinaryMask(np.asarray(temp_edge, dtype=np.float32))
