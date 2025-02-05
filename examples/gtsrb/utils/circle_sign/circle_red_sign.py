from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont
from examples.gtsrb.utils.circle_sign.circle_utils import ImageModifier, run_detection
from examples.gtsrb.utils.deprecated_speed_sign.operation.write_speed_operation import _write_on_image
from examples.gtsrb.utils.domain.image.rgb_image import RGBImage


class ImageAnalyzer:
    """
    This function finds the number of pixels in the rim of the circle. Note that the rim here is the part of image which lies in the
    co-centric circle extending from 0.7*radius to the actual radius of the circle, where the center of both the circles is defined by
    center_x and center_y
    """

    @staticmethod
    def extract_circle_pixels(image, circle):
        if not circle:
            return image

        scaled_radius = int(circle[2] * 0.7)
        center = (circle[0], circle[1])
        mask = np.zeros_like(image)
        cv2.circle(mask, center, circle[2], (255, 255, 255), thickness=-1)
        cv2.circle(mask, center, scaled_radius, (0, 0, 0), thickness=-1)
        pixels_in_rim = np.count_nonzero(mask) // 3
        circle_image = cv2.bitwise_and(image, mask)

        return circle_image, pixels_in_rim

    '''Counts the number of pixels in the rim of the image which lies in the red color range'''

    @staticmethod
    def count_pixels_in_range(cropped_image, lower_bound, upper_bound):
        mask = cv2.inRange(cropped_image, np.array(lower_bound), np.array(upper_bound))
        pixels_in_range = np.count_nonzero(mask)

        return pixels_in_range

    '''
    Gives the confidence value for the cirlce detected in the image. This is given by:
    no_of_pixels_in_red_color_range/total_no_of_pixel
    '''

    def confidence(self, image, circle):
        if not circle:
            return 0

        crop_image, pixels_in_rim = self.extract_circle_pixels(image, circle)
        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_RGB2HSV)
        # cv2.imshow('img', crop_image)
        # cv2.waitKey(0)
        li_range = [
            [
                [0, 100, 20],  # red color range 1 - lower bound
                [10, 255, 255]  # red color range 1 - upper bound
            ],
            [
                [170, 100, 20],  # red color range 2 - lower bound
                [180, 255, 255]  # red color range 2 - upper bound
            ],
        ]
        valid_pixels = 0

        for i in li_range:
            num = self.count_pixels_in_range(crop_image, i[0], i[1])
            valid_pixels += num

        return (valid_pixels / (pixels_in_rim)) * 100


'''Writes text on the image'''


def write_text_in_circle(image, circle_info, text, radius_factor=1.0, font=cv2.FONT_HERSHEY_SIMPLEX):
    center_x, center_y, radius = circle_info
    radius = int(radius * radius_factor)
    font_scale = max(radius / 40.0, 0.5)
    font_thickness = max(int(radius / 20.0), 1)

    if (radius / 10.0) >= 1.6:
        font_scale = radius / 20.0
        font_thickness = int(radius / 8.75)

    result_image = image.copy()
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_position = (int(center_x - text_size[0] / 2), int(center_y + text_size[1] / 2))
    cv2.putText(result_image, text, text_position, font, font_scale, (0, 0, 0), font_thickness,
                lineType=cv2.LINE_AA)

    return result_image


CANNY_HED = ImageAnalyzer()
IMAGE_MODIFIER = ImageModifier()


def calc_font_size(box, path_to_font, text, font_start_size):
    font_size = font_start_size
    path = Path(__file__).parent.parent.parent.parent.parent
    path_to_font = path_to_font[2:]
    size = None
    while (
            size is None or size[0] > box[2] - box[0] or
            size[1] > box[3] - box[1]
    ) and font_size > 0:
        font = ImageFont.truetype(path / path_to_font, font_size)
        # size = font.getsize(text)
        _, _, *size = font.getbbox(text)
        font_size -= 1
    return ImageFont.truetype(path / path_to_font, font_size)


def run_red_circle_processing(input_image, text):
    image = input_image
    list_circle = run_detection(image)
    max_confidence = 0
    max_circle = []
    for i in list_circle:
        temp = CANNY_HED.confidence(image, i)
        if temp > max_confidence:
            max_confidence = temp
            max_circle = i

    if max_confidence < 50:
        return None

    blank_image = IMAGE_MODIFIER.extraction_color(image.copy(), max_circle, 0.7)
    final_image = write_text_in_circle(blank_image, max_circle, text, 0.49,
                                       font=cv2.FONT_HERSHEY_SIMPLEX)
    return final_image


def run_red_circle_processing_realistic_font(input_image, text, font_path):
    image = input_image
    list_circle = run_detection(image)
    max_confidence = 0
    max_circle = []
    for i in list_circle:
        temp = CANNY_HED.confidence(image, i)
        if temp > max_confidence:
            max_confidence = temp
            max_circle = i

    if max_confidence < 50:
        return None

    blank_image = IMAGE_MODIFIER.extraction_color(image.copy(), max_circle, 0.7)
    center_x, center_y, radius = max_circle
    radius = 0.6 * radius
    bounding_box = (
        center_x - radius,
        center_x - radius,
        center_y + radius,
        center_y + radius
    )
    font_start_size = int(min(image[:, :, 0].shape) / 2)
    font = calc_font_size(bounding_box, font_path, text, font_start_size)
    final_image = _write_on_image(
        RGBImage(blank_image), bounding_box, font, text
    )
    return final_image.value
