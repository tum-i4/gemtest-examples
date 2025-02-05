import typing

from examples.gtsrb.utils.domain.image.binary_mask import BinaryMask
from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.domain.image.rgba_image import RGBAImage

_Image = typing.Union[RGBImage, RGBAImage, BinaryMask]


class ProcessingStep:
    """Wrapper that asserts type safety on transformation steps."""

    def __init__(self, name: typing.Optional[str] = None):
        if name is None:
            self._name = self.__class__.__name__
        else:
            self._name = name

    def apply(self, **kwargs) -> typing.Any:
        """Apply a transformation operation."""
        raise NotImplementedError

    def __str__(self) -> str:
        return self._name

    __repr__ = __str__


class ImageProcessingStep(ProcessingStep):
    """Wrapper that asserts type safety on visual transformation steps."""

    def apply(self, fn_input: _Image) -> typing.Optional[_Image]:
        """Apply an image transformation operation on a copy of the image."""
        raise NotImplementedError

    def __str__(self) -> str:
        return self._name

    __repr__ = __str__
