from typing import Optional, Union

from examples.gtsrb.utils.domain.image.binary_mask import BinaryMask
from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.domain.image.rgba_image import RGBAImage
from examples.gtsrb.utils.processing_steps.processing_step import ImageProcessingStep

_Image = Union[RGBImage, RGBAImage, BinaryMask]


class FlipApplyStickerOperation(ImageProcessingStep):

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def apply(self, fn_input: _Image) -> Optional[_Image]:
        pass
