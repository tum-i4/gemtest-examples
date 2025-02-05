from pathlib import Path

import cv2
import numpy as np


class CropLayer(object):
    """Used as steup for the HED edge detector"""

    def __init__(self, params, blobs):
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):  # method name must not be refactored!
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.startY:self.endY, self.startX:self.endX]]


class ImageProcessor:
    def __init__(self, proto_path, model_path):
        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        cv2.dnn_registerLayer("Crop", CropLayer)

    '''Used for applying the HED edge detectr to the image'''

    def f_hed(self, image):
        gray = cv2.equalizeHist(image)

        (H, W) = image.shape[:2]
        gray_3channel = cv2.merge([gray, gray, gray])
        blob = cv2.dnn.blobFromImage(
            gray_3channel, scalefactor=1.0, size=(W, H),
            mean=(104.00698793, 116.66876762, 122.67891434),
            swapRB=False, crop=False
        )

        self.net.setInput(blob)
        hed = self.net.forward()
        hed = cv2.resize(hed[0, 0], (W, H))
        hed = (255 * hed).astype("uint8")

        return hed

    '''Used for applying the canny edge detectr to the image'''

    @staticmethod
    def f_canny(image):
        gray = cv2.equalizeHist(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blurred, 30, 150)

        return canny


class ImageModifier:
    @staticmethod
    def extraction_color(image, circle, factor=1.0):
        center_x, center_y, radius = circle
        radius = int(radius * factor)
        # Create a mask with white pixels inside the circle
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), thickness=cv2.FILLED)

        # Use bitwise_and to set the pixels inside the circle to white in the original image
        result_image = cv2.bitwise_and(image, mask)

        lower_bound = np.array([75, 0, 99], dtype=np.uint8)
        upper_bound = np.array([179, 62, 255], dtype=np.uint8)

        # Creating a binary mask using inRange function
        img_threshold = cv2.inRange(result_image, lower_bound, upper_bound)
        img_inverse_mask = cv2.bitwise_not(img_threshold)
        img_pixels = cv2.bitwise_and(result_image, result_image, mask=img_inverse_mask)
        img_pixels = img_pixels.reshape((-1, 3))
        img_pixels = img_pixels[img_pixels[:, 2] > 0]
        average_color = np.mean(img_pixels, axis=0)

        modified_image = image.copy()

        # Get indices of pixels inside the mask
        indices = np.where(mask != 0)

        # Assign average_color to each pixel inside the mask individually
        modified_image[indices[0], indices[1], :] = average_color

        return modified_image


class CircleDetector:
    """
    Used for detecting circle from the images obtained from the edge detector using Hough Detector and if a circle
    is detected then it returns a list which is of the form [center_x , center_y , radius] to define the circle on
    the image.
    """

    @staticmethod
    def detect(edge_image):
        (H, W) = edge_image.shape[:2]
        temp = min(H, W)
        circles = cv2.HoughCircles(edge_image, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=30, param2=20,
                                   minRadius=10,
                                   maxRadius=temp // 2)

        det = 0

        if circles is not None:
            circles = np.uint16(np.around(circles))

        ret_list = []

        if circles is not None:
            ret_list.append(circles[0][0][0])
            ret_list.append(circles[0][0][1])
            ret_list.append(circles[0][0][2])

        return ret_list


CIRCLE_DETECTOR = CircleDetector()
parent_dir = Path(__file__).parent
IMAGE_PROCESSOR = ImageProcessor(str(parent_dir) + '/deploy.prototxt.txt',
                                 str(parent_dir) + '/hed_pretrained_bsds.caffemodel')


def run_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    normalized_gray = cv2.equalizeHist(gray)
    image_processor = IMAGE_PROCESSOR
    ori = normalized_gray.copy()
    canny = normalized_gray.copy()
    canny = image_processor.f_canny(canny)
    hed = normalized_gray.copy()
    hed = image_processor.f_hed(hed)
    hed_norm = cv2.equalizeHist(hed)
    circle_detector = CIRCLE_DETECTOR
    circle_orig = circle_detector.detect(ori)
    circle_canny = circle_detector.detect(canny)
    circle_hed = circle_detector.detect(hed)
    circle_hed_norm = circle_detector.detect(hed_norm)

    list_circle = [circle_orig, circle_canny, circle_hed, circle_hed_norm]
    return list_circle
