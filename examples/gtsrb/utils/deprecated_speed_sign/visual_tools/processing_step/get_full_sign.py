import typing

from examples.gtsrb.utils.domain.image.binary_mask import BinaryMask
from examples.gtsrb.utils.processing_steps.processing_step import ImageProcessingStep


class GetFullSign(ImageProcessingStep):
    """Get all pixels belonging to a driving sign.

    This is simply a bitwise not of the pixels in the background.
    The argument in apply is thus expected to be a mask selecting
    the background from a past step.
    """

    def apply(
            self, fn_input: BinaryMask
    ) -> typing.Optional[BinaryMask]:
        return fn_input.negate()
