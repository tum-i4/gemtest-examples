import copy
from typing import List, Tuple

import cv2
import numpy as np

from examples.gtsrb.utils.domain.image.binary_mask import BinaryMask
from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.processing_steps.processing_step import (ImageProcessingStep,
                                                                   _Image, ProcessingStep)
from examples.gtsrb.utils.triangle_sign.triangle_error import TriangleDetectionError


class FindTriangleContour(ProcessingStep):
    """Finds the largest triangle contour in the image and returns the contour if one is present"""

    @staticmethod
    def is_triangle(c) -> bool:
        # approximate the curve of the contour. Returns true if the curve consists of 3 vertices else false
        triangle_discovered = False
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 3:
            triangle_discovered = True
        return triangle_discovered

    @staticmethod
    def grab_contours(cnts) -> np.ndarray:
        # if the length the contours tuple returned by cv2.findContours
        # is '2' then we are using either OpenCV v2.4, v4-beta, or
        # v4-official
        if len(cnts) == 2:
            cnts = cnts[0]

        # if the length of the contours tuple is '3' then we are using
        # either OpenCV v3, v4-pre, or v4-alpha
        elif len(cnts) == 3:
            cnts = cnts[1]

        # return the actual contours array
        return cnts

    def apply(self, mask: BinaryMask) -> np.ndarray:
        # finds triangle contours in the threshold image and return the largest contour if one is found
        cnts = cv2.findContours(mask.value.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = self.grab_contours(cnts)
        for c in sorted(cnts, key=len, reverse=True):
            if len(c) <= 20:
                raise TriangleDetectionError(f"{self._name}: Contour not large enough.")
            # compute the moments of the contour, then detect the name of the
            # shape using only the contour
            M = cv2.moments(c)
            if all([x != 0 for x in [M["m10"], M["m00"], M["m01"], M["m01"]]]):
                if self.is_triangle(c):
                    return c
                raise TriangleDetectionError(f"{self._name}: No triangle contour found.")
            raise TriangleDetectionError(f"{self._name}: No triangle contour found.")


class ImageFromContour(ImageProcessingStep):
    """Cuts out the part of an image defined by a contour and returns the cutout"""

    def __init__(self, contour):
        self.contour = contour
        super().__init__()

    def apply(self, fn_input: RGBImage) -> RGBImage:
        image = fn_input.value
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.contour], 255)
        dst = cv2.bitwise_and(image, image, mask=mask)
        return RGBImage(dst)


class TriangleCornerExtractor(ProcessingStep):
    """Extracts the corners of a triangle from a provided triangle contour"""

    def apply(self, fn_input: _Image) -> np.ndarray:
        img = copy.deepcopy(fn_input.value)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(image=img, maxCorners=3, qualityLevel=0.01, minDistance=15)
        corners = np.int32(corners)
        final_corners = [tuple(corner[0]) for corner in corners]
        final_corners.sort()
        return np.float32(final_corners)


class CalcTriangleCentroid(ProcessingStep):
    """Calculates the centroid of a triangle from a contour"""

    def apply(self, contour) -> Tuple[int, int]:
        M = cv2.moments(contour)
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))
        return cX, cY


class TriangleCenterCheck(ProcessingStep):
    """Checks if the centroids of the both contour as within a distance with each other smaller than the threshold.
    Returns the centroid of the first contour if the distance is below the threshold."""

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        super().__init__()

    def apply(self, contour1: np.ndarray, contour2: np.ndarray) -> Tuple[int, int]:
        cX1, cY1 = CalcTriangleCentroid().apply(contour1)
        cX2, cY2 = CalcTriangleCentroid().apply(contour2)
        dx = (cX1 - cX2) ** 2
        dy = (cY1 - cY2) ** 2
        if (dx + dy) > (self.threshold ** 2):
            raise TriangleDetectionError(f"{self._name}: Distance between triangle centroids is too big.")
        else:
            return cX1, cY1


class ExtractBorderPixels(ProcessingStep):
    """ Returns a list of all pixels which are not 0 but have neighbouring pixels that are 0."""

    def apply(self, fn_input: _Image) -> List:
        img = fn_input.value
        k_size = 5
        half_k = int((k_size - 1) / 2)

        valid_indices = np.stack(np.nonzero(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)), axis=-1)

        pixel_list = []
        for i, j in valid_indices:
            neighbours = img[i - half_k:i + half_k + 1, j - half_k:j + half_k + 1]
            are_all_pixels_not_zero = neighbours.all()
            is_any_pixel_not_zero = neighbours.any()
            if is_any_pixel_not_zero and not are_all_pixels_not_zero:
                pixel_list.append((i, j))

        return list(np.unique(pixel_list, axis=0))


class AdaptedBorderMask(ImageProcessingStep):
    """ Applies an adaptive threshold to an image and returns a BinaryMask of all Pixels unequal to 0."""

    def __init__(self, invert: bool = False):
        super().__init__()
        self.invert = invert

    def apply(self, fn_input: RGBImage) -> BinaryMask:
        img = copy.deepcopy(fn_input.value)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.adaptiveThreshold(img, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                    thresholdType=cv2.THRESH_BINARY, blockSize=11, C=0)
        c = FindTriangleContour().apply(BinaryMask(img))
        mask = np.zeros(img.shape[:2], dtype="uint8")
        mask = cv2.fillPoly(mask, [c], 255)
        if self.invert:
            mask = cv2.inRange(mask, 0, 1)
        return BinaryMask(mask)


class BoxFilter(ImageProcessingStep):
    """ A custom implementation of a box filter which applies the filter only to the pixels provided in the
    parameter pixel_list."""

    def __init__(self, pixel_list: List[Tuple[int, int]]):
        super().__init__()
        self.pixel_list = pixel_list

    def apply(self, fn_input: RGBImage) -> RGBImage:
        img = fn_input.value
        img = copy.deepcopy(img)
        # define kernel size
        # (defines which area is used to calculate the new value for the pixel in the center of the kernel)
        k_size = 3
        half_k = int((k_size - 1) / 2)
        # define neighbour area
        # defines to which area around the original pixel the box filter is also applied.
        n_size = 3
        half_n = int((n_size - 1) / 2)
        kernel = np.ones((k_size, k_size, 1)) / (k_size * k_size)
        # iterate through all points of the provided list
        for x, y in self.pixel_list:
            # apply the box filter to all neighbouring pixels(as defined by n_size) and the original pixel
            for i in range(x - half_n, x + half_n):
                for j in range(y - half_n, y + half_n):
                    # calculate the average pixel value from the pixels in the kernel
                    # and set this value for (i,j) in the image
                    img[i, j, :] = np.sum(img[i - half_k:i + half_k + 1, j - half_k:j + half_k + 1, :] * kernel,
                                          axis=(0, 1))
        return RGBImage(img)
