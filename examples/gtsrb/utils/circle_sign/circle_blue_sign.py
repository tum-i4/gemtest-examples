import cv2
import numpy as np

from examples.gtsrb.utils.circle_sign.circle_utils import ImageModifier, run_detection


class ImageAnalysis:

    @staticmethod
    def cropped(image, circle):
        if len(circle) != 3:
            return image

        center_x, center_y, radius = circle[0], circle[1], circle[2]
        mask = np.zeros_like(image)
        # Draw a filled white circle on the mask
        cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), thickness=cv2.FILLED)
        # Use the mask to extract pixels from the original image
        result_image = cv2.bitwise_and(image, mask)
        return result_image

    @staticmethod
    def detect(original_image, edge_image):
        # Use Hough Circle Transform to detect circles in the edge image
        (H, W) = original_image.shape[:2]
        temp = min(H, W)
        circles = cv2.HoughCircles(edge_image, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=30, param2=20,
                                   minRadius=10, maxRadius=temp // 2)

        if circles is not None:
            # Convert circle parameters to integer
            circles = np.uint16(np.around(circles))

        ret_list = []
        if circles is not None:
            ret_list.append(circles[0][0][0])
            ret_list.append(circles[0][0][1])
            ret_list.append(circles[0][0][2])

        return ret_list

    def confidence(self, image, circle):
        cropped_image = self.cropped(image, circle)
        total_pixels = np.count_nonzero(cropped_image) // 3
        lower_bound = [100, 150, 0]
        upper_bound = [140, 255, 255]
        hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_image, np.array(lower_bound), np.array(upper_bound))
        inv_mask = cv2.bitwise_not(mask)
        cv2.bitwise_and(cropped_image, cropped_image, mask=inv_mask)
        # Count the number of pixels in the specified HSV range
        pixels_in_range = np.count_nonzero(mask)
        total_confidence = pixels_in_range / total_pixels
        return total_confidence


def overlay_image_within_circle(background, overlay, circle, target_brightness=127):
    center_x, center_y, radius = circle[0], circle[1], circle[2]
    mask = np.zeros_like(background)
    cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), thickness=cv2.FILLED)
    overlay = cv2.resize(overlay, (2 * radius, 2 * radius))
    roi = cv2.bitwise_and(background, mask)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_brightness_background = np.mean(gray_roi)
    brightness_ratio = mean_brightness_background / target_brightness
    overlay = np.clip((overlay * brightness_ratio), 0, 255).astype(np.uint8)
    x, y = center_x - radius, center_y - radius
    roi[y:y + 2 * radius, x:x + 2 * radius] = overlay
    result = cv2.add(background, roi)

    return result


CANNY_HED = ImageAnalysis()
IMAGE_MODIFIER = ImageModifier()


def run_blue_circle_processing(input_image, overlay, confidence_thr=0.7):
    list_circle = run_detection(input_image)

    best_circle = []
    max_confidence = 0
    for i in list_circle:
        temp = CANNY_HED.confidence(input_image.copy(), i)
        if temp > max_confidence:
            max_confidence = temp
            best_circle = i

    if max_confidence < confidence_thr:
        return None

    blank = IMAGE_MODIFIER.extraction_color(input_image.copy(), best_circle)
    final_image = overlay_image_within_circle(blank, overlay, best_circle)

    return final_image
