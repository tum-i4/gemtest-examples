import typing

from examples.gtsrb.utils.domain.image.binary_mask import BinaryMask
from examples.gtsrb.utils.processing_steps.processing_step import ImageProcessingStep


class SelectRegionInsideBorder(ImageProcessingStep):
    """Gets the inner part of the driving sign."""

    def __init__(
            self,
            border_mask: BinaryMask,
            name: typing.Optional[str] = None
    ):
        super().__init__(name)
        self._border_mask = border_mask

    def apply(self, fn_input: BinaryMask) -> BinaryMask:
        return fn_input.xor(self._border_mask)
