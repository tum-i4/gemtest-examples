import typing

from examples.gtsrb.utils.domain.image.binary_mask import BinaryMask
from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.processing_steps.processing_step import ImageProcessingStep

Image = typing.Union[RGBImage, BinaryMask]


class ProcessingStepSequence:
    """Applies a series of VisualProcessingStep objects consecutively.

    The pipeline will immediately return if any step returns None,
    signifying a failure of one of the steps.
    """

    def __init__(self, funcs: typing.Sequence[ImageProcessingStep]):
        self.funcs = funcs

    def apply(self, img: Image) -> Image:
        copy = img.copy()
        for fn in self.funcs:
            copy = fn.apply(copy)
            if copy is None:
                return None
        return copy

    def extend(self, *new_fns: ImageProcessingStep) -> 'ProcessingStepSequence':
        """Add a new processing step and return a new pipeline."""
        fns = [*self.funcs, *new_fns]
        return ProcessingStepSequence(fns)
