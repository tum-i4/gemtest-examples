import typing

from examples.gtsrb.utils.deprecated_speed_sign.visual_tools.processing_step.get_full_sign import \
    GetFullSign
from examples.gtsrb.utils.deprecated_speed_sign.visual_tools.processing_step.get_sign_background import \
    GetSignBackground
from examples.gtsrb.utils.deprecated_speed_sign.visual_tools.processing_step.largest_component_from_mask \
    import LargestComponentFromMask
from examples.gtsrb.utils.deprecated_speed_sign.visual_tools.processing_step.validate_full_sign_mask import \
    ValidateFullSignMaskOptions, ValidateFullSignMask
from examples.gtsrb.utils.domain.color.hsv_color import HSVColor
from examples.gtsrb.utils.domain.image.binary_mask import BinaryMask
from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.processing_steps.binary_mask_from_hsv_color import \
    BinaryMaskFromHSVColor
from examples.gtsrb.utils.processing_steps.processing_step_sequence import \
    ProcessingStepSequence

Image = typing.Union[RGBImage, BinaryMask]


def find_sign_by_border_color(
        colors: typing.Tuple[HSVColor],
        validate_options: typing.Optional[
            ValidateFullSignMaskOptions
        ]
) -> typing.Tuple[
    ProcessingStepSequence,
    ProcessingStepSequence,
    ProcessingStepSequence
]:
    """Returns pipelines for extracting the border, background and full sign
    from RGB image. Ech pipeline returns a BinaryMask that can be used to
    select the corresponding pixels

    This procedure assumes that the input color (denoted with plural
    [colors] as multiple HSV intervals can be selected) is found
    overwhelmingly in the sign border, and it will discard all
    regions matching the filter except the biggest one.

    Returns:
        A tuple containing a pipeline for getting the border of the sign mask,
        a pipeline for getting the background of the image mask and a pipeline
        for getting the full sign mask
    """
    border_mask_pipeline = ProcessingStepSequence([BinaryMaskFromHSVColor(colors), LargestComponentFromMask()])

    background_mask_pipeline = border_mask_pipeline.extend(
        GetSignBackground()
    )

    full_sign_mask_pipeline = background_mask_pipeline.extend(
        GetFullSign(),
    )
    if validate_options is not None:
        full_sign_mask_pipeline = full_sign_mask_pipeline.extend(
            ValidateFullSignMask(validate_options)
        )

    return border_mask_pipeline, background_mask_pipeline, full_sign_mask_pipeline
