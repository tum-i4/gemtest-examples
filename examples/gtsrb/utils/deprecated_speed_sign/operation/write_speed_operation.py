import math
import typing

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import gemtest as gmt
from examples.gtsrb.utils.deprecated_speed_sign.visual_tools.binary_mask_utils import \
    get_points_centroid
from examples.gtsrb.utils.deprecated_speed_sign.visual_tools.perspective_utils \
    import PerspectiveTransformFn, calculate_perspective_distortion
from examples.gtsrb.utils.deprecated_speed_sign.visual_tools.processing_step import \
    validate_full_sign_mask
from examples.gtsrb.utils.deprecated_speed_sign.visual_tools.processing_step \
    .find_sign_by_border_color_sequence import find_sign_by_border_color
from examples.gtsrb.utils.domain import point_2d
from examples.gtsrb.utils.domain.color.hsv_color import COLOR_RANGES
from examples.gtsrb.utils.domain.image.binary_mask import BinaryMask
from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.processing_steps.processing_step import ImageProcessingStep

ImageFontType = typing.Type['ImageFont']
# BoundingBox = namedtuple("BoundingBox",
#                          "top_left_x top_left_y bottom_right_x bottom_right_yy")
BoundingBox = typing.Tuple[int, int, int, int]


def _get_largest_inscribed_square(
        points: BinaryMask
) -> typing.Iterable[point_2d.Point2D]:
    """Get coordinates of the biggest square that can be inscribed inside
    cloud of points.

    The computation starts from the centroid and binary searches for the
    largest side length that keeps all vertices within the cloud of points.

    Coordinates are returned in order
    top_left, top_right, bottom_left, bottom_right
    """

    def vertices_inside_points(
            _points: np.ndarray, _y_top: int, _y_down: int,
            _x_left: int, _x_right: int
    ) -> bool:
        if _points[_y_top][_x_left] == BinaryMask.UNSELECTED_PIXEL_VALUE:
            # Left upper corner is outside
            return False
        if _points[_y_top][_x_right] == BinaryMask.UNSELECTED_PIXEL_VALUE:
            # Right upper corner is outside
            return False
        if _points[_y_down][_x_left] == BinaryMask.UNSELECTED_PIXEL_VALUE:
            # Left down corner is outside
            return False
        if _points[_y_down][_x_right] == BinaryMask.UNSELECTED_PIXEL_VALUE:
            # Right down corner is outside
            return False
        return True

    def get_vertices_coords_for_length(
            _c_y: int, _c_x: int, trial_len: int
    ) -> BoundingBox:
        """
        Calculate coordinates of square by drawing a vertical imaginary
        segment whose middle is in the center and calculating the coordinates
        as symmetric points from its end.

        Arguments:
            _c_y: Y coordinate of the center of the square
            _c_x: X coordinate of the center of the square
            trial_len: The length of the square
        """
        _y_top = _c_y - trial_len // 2
        _y_down = _c_y + trial_len // 2
        _x_left = _c_x - trial_len // 2
        _x_right = _c_x + trial_len // 2
        return _y_top, _y_down, _x_left, _x_right

    center = get_points_centroid(points.value)
    length_upper_bound = int(0.4 * min(points.value.shape[0], points.value.shape[1]))
    best_y_top, best_y_down, best_x_left, best_x_right = 0, 0, 0, 0
    # Binary search the length of the square that keeps us inside the points
    left_end_interval, right_end_interval = 0, length_upper_bound
    while left_end_interval < right_end_interval:
        trial_length = (left_end_interval + right_end_interval) // 2
        y_top, y_down, x_left, x_right = get_vertices_coords_for_length(
            center.y, center.x, trial_length
        )
        if vertices_inside_points(
                points.value, y_top, y_down, x_left, x_right
        ):
            # Try bigger lengths
            left_end_interval = trial_length + 1
            best_y_top = y_top
            best_y_down = y_down
            best_x_left = x_left
            best_x_right = x_right
        else:
            # Too large, try smaller lengths
            right_end_interval = trial_length - 1
    # Bias introduction - make the bounding box larger than then
    # inscribed square
    _BB_BIAS = 5
    best_y_top -= _BB_BIAS
    best_x_left -= _BB_BIAS
    best_y_down += _BB_BIAS
    best_x_right += _BB_BIAS
    # Group coordinates into vertices for readability; they are in (Y, X) format
    top_left = (best_y_top, best_x_left)
    top_right = (best_y_top, best_x_right)
    bottom_left = (best_y_down, best_x_left)
    bottom_right = (best_y_down, best_x_right)
    return [
        point_2d.Point2D(x=top_left[1], y=top_left[0]),
        point_2d.Point2D(x=top_right[1], y=top_right[0]),
        point_2d.Point2D(x=bottom_left[1], y=bottom_left[0]),
        point_2d.Point2D(x=bottom_right[1], y=bottom_right[1])
    ]


def _calculate_font_size(
        mask: BinaryMask, text: str, path_to_font: str
) -> typing.Tuple[BoundingBox, ImageFontType]:
    """Calculates the largest font size for which input text fits inside
    the input mask.

    Returns:
        The best bounding box and font for given text.
    """
    top_left, top_right, bottom_left, bottom_right = (
        _get_largest_inscribed_square(mask)
    )
    # Calculate the largest font size that fits given bounding box
    box = (top_left.x, top_left.y, bottom_right.x, bottom_right.y)
    font_size = 500
    text_width = None
    text_height = None
    # todo: search for font size with binary search
    while (
            (text_width is None) or
            (text_height is None) or
            (text_width > box[2] - box[0]) or
            (text_height > box[3] - box[1])
    ) and font_size > 0:
        font = ImageFont.truetype(path_to_font, font_size)
        left, top, right, bottom = font.getbbox(text)
        text_width = right - left
        text_height = bottom - top
        font_size -= 1
    return box, ImageFont.truetype(path_to_font, font_size)


def _write_on_image(
        rgb: RGBImage,
        bounding_box: BoundingBox,
        font: ImageFontType,
        text: str
) -> RGBImage:
    """Write text on copy of an image at input coordinates and with input font size."""
    copy: np.ndarray = rgb.value.copy()
    pil_img = Image.fromarray(copy)
    draw = ImageDraw.Draw(pil_img)
    draw.multiline_text((bounding_box[0], bounding_box[1]), text, (0, 0, 0), font)
    # PyCharm panics about np.array conversion
    # noinspection PyTypeChecker
    return RGBImage(np.array(pil_img))


def _is_valid_bounding_box(
        bounding_box: BoundingBox,
        img_centroid: point_2d.Point2D,
        image_size: typing.Tuple[int, int],
        tolerance_range: float,
        perspective_op_transform: PerspectiveTransformFn
) -> bool:
    """Checks if the bounding box of the written text shifts
    too much from the image center and drops it if needed.

    This is achieved by comparing the centroid of the bounding box
    shifted back to original perspective with the center of the
    image from the original perspective. This builds upon the fact
    that the samples from the dataset tend to be centered.

    Returns:
        True if the image adheres to the quality standard.
    """

    def _euclidean(p1: point_2d.Point2D, p2: point_2d.Point2D) -> float:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    x_top_left, y_top_left, x_bottom_right, y_bottom_right = bounding_box
    img_mask = np.ndarray(image_size, np.float32)
    img_mask.fill(BinaryMask.UNSELECTED_PIXEL_VALUE)
    img_mask[y_top_left:y_bottom_right + 1, x_top_left:x_bottom_right] = \
        BinaryMask.SELECTED_PIXEL_VALUE
    bm = BinaryMask(img_mask)
    original_perspective_bm = BinaryMask(
        perspective_op_transform(bm.value)
    )
    warped_centroid = get_points_centroid(
        original_perspective_bm.value
    )
    # (224, 224)
    distance = _euclidean(warped_centroid, img_centroid)
    percentual_threshold = (tolerance_range / 224) * image_size[0]
    return distance <= percentual_threshold


class WriteSpeedOperation(ImageProcessingStep):
    # _TOLERANCE_RANGE_BBOX = 7 as submitted in practicum ws22/23
    _TOLERANCE_RANGE_BBOX = 20

    def __init__(
            self, speed_text: str, font_path: str,
            control_image_path: str, name: typing.Optional[str] = None
    ):
        super().__init__(name)
        self._speed_text = speed_text
        self._font_path = font_path
        self._control_img = RGBImage.read_from_disk(control_image_path)

    def apply(
            self, fn_input: RGBImage
    ) -> typing.Optional[RGBImage]:
        _, _, full_sign_mask_pipeline = \
            find_sign_by_border_color(
                COLOR_RANGES.RED_COLOR.value,
                validate_full_sign_mask.ValidateFullSignMaskOptions(
                    min_area_selected_pipeline=0.1,
                    max_area_selected_pipeline=0.8,
                    tolerated_displacement_from_center=0.1,
                    search_range_gaps_inside_center=0.1,
                )
            )
        self._control_img.value = cv2.resize(
            self._control_img.value,
            dsize=fn_input.value.shape[:2][::-1],
            interpolation=cv2.INTER_LANCZOS4
        )
        rgb_mask: typing.Optional[BinaryMask] = \
            full_sign_mask_pipeline.apply(fn_input)
        control_mask: typing.Optional[BinaryMask] = \
            full_sign_mask_pipeline.apply(self._control_img)

        if rgb_mask is None or control_mask is None:
            gmt.skip("RGB or control mask is None")

        to_control_perspective, from_control_perspective = \
            calculate_perspective_distortion(
                rgb_mask, control_mask
            )

        rgb_sign_crop = fn_input.get_crop(rgb_mask)
        normalised_rgb_sign_crop = RGBImage(
            to_control_perspective(rgb_sign_crop.value)
        )
        # Abbreviation for the variable above
        nrsc_full_mask = full_sign_mask_pipeline.apply(normalised_rgb_sign_crop)
        if nrsc_full_mask is None:
            gmt.skip("No full sign mask found")

        bounding_box, font = _calculate_font_size(
            nrsc_full_mask, self._speed_text, self._font_path
        )
        if not _is_valid_bounding_box(
                bounding_box,
                img_centroid=point_2d.Point2D(x=fn_input.value.shape[:2][::-1][0] / 2,
                                              y=fn_input.value.shape[:2][::-1][1] / 2),
                image_size=fn_input.value.shape[:2][::-1],
                tolerance_range=WriteSpeedOperation._TOLERANCE_RANGE_BBOX,
                perspective_op_transform=from_control_perspective
        ):
            gmt.skip("Bounding box too far from center")

        rgb_sign_crop_with_speed = _write_on_image(
            normalised_rgb_sign_crop, bounding_box, font, self._speed_text
        )
        final_result = RGBImage(
            from_control_perspective(rgb_sign_crop_with_speed.value)
        )
        # todo:
        # now overlay the font over the fn_input
        # transform the text with the perspective transform and paste over fn_input

        final_result_mask: BinaryMask = \
            full_sign_mask_pipeline.apply(final_result)
        if final_result_mask is None:
            gmt.skip("Final result mask is none")

        eroded_mask = final_result_mask.erode()
        bg_init_img_mask = eroded_mask.negate()
        bg_init_img = fn_input.get_crop(bg_init_img_mask)
        return bg_init_img.overlay(final_result)
