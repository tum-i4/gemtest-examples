import glob
import os.path
import random
from typing import Optional, Union, Tuple

import numpy as np

from examples.gtsrb.utils.domain.image.rgb_image import RGBImage
from examples.gtsrb.utils.domain.image.rgba_image import RGBAImage
from examples.gtsrb.utils.processing_steps.overlay import Overlay
from examples.gtsrb.utils.processing_steps.processing_step \
    import ImageProcessingStep
from examples.gtsrb.utils.processing_steps.rotate_and_rescale import RotateAndRescale


def _find_window(img: RGBAImage,
                 window_width: int,
                 window_height: int,
                 min_nontransparent_ratio: float = 0,
                 max_nontransparent_ratio: float = 1,
                 # Set to >0 to ensure are no abrupt cuts at the bottom of the image
                 min_nontransparent_ratio_bottom: float = 0,
                 max_iter: int = 50,
                 never_return_none: bool = True) -> Optional[np.ndarray]:
    """
    Find and cut a window from img with a specified ratio of nontransparent pixels
    throughout the image and a specified ratio of nontransparent pixels in the bottom row.
    Nontransparent pixels are pixels with alpha > 0. Try at most max_iter random samples and
    return the window which satisfies the requirements best
    """
    img: np.ndarray = img.value
    assert img.shape[2] == 4  # ensure img has an alpha channel
    (h, w) = img.shape[:2]
    #
    # if h < window_width or w < window_height:  # if img is smaller than window
    #     scale = max(window_height / img.shape[0], window_width / img.shape[1])
    #     img = _rotate_and_rescale(img, scale=scale)
    #     (h, w) = img.shape[:2]

    min_so_far = 1.0
    window_min = None
    max_so_far = 0.0
    window_max = None

    # choose strategy for randomly generating y and x (coordinates of the top left corner of
    # the window)
    def next_y():
        return random.randrange(-window_height, h - window_height)

    def next_x():
        return random.randrange(-window_width, w - window_width)

    if h < window_height:
        def next_y():
            return random.randrange(-h, h)
    if w < window_width:
        def next_x():
            return random.randrange(-w, w)

    curr_bottom_nontransparent = 0
    y, x = next_y(), next_x()

    for _ in range(max_iter):
        y, x = next_y(), next_x()
        curr_window = img[max(y, 0):y + window_height, max(x, 0):x + window_width, :]
        if y < 0:
            curr_window = np.pad(curr_window, (
                (-y, 0),
                (0, 0),
                (0, 0)
            ))
        if x < 0:
            curr_window = np.pad(curr_window, (
                (0, 0),
                (-x, 0),
                (0, 0)
            ))
        if curr_window.shape != (window_height, window_width, 4):
            curr_window = np.pad(curr_window, (
                (0, abs(window_height - curr_window.shape[0])),
                (0, abs(window_width - curr_window.shape[1])),
                (0, 0)
            ))

        # select only alpha values and check how many nontransparent pixels there are in the
        # window
        alphas = curr_window[:, :, 3]

        curr_bottom_nontransparent = 0
        if min_nontransparent_ratio_bottom > 0:
            curr_bottom_nontransparent = np.count_nonzero(alphas[-1, :]) / float(window_width)
        curr_nontransparent = np.count_nonzero(alphas) / (window_height * window_width)

        if min_nontransparent_ratio <= curr_nontransparent <= max_nontransparent_ratio and \
                curr_bottom_nontransparent >= min_nontransparent_ratio_bottom:
            return curr_window
        elif min_nontransparent_ratio > curr_nontransparent >= max_so_far:
            max_so_far = curr_nontransparent
            window_max = curr_window
        elif max_nontransparent_ratio < curr_nontransparent <= min_so_far:
            min_so_far = curr_nontransparent
            window_min = curr_window

    if curr_bottom_nontransparent < min_nontransparent_ratio_bottom:
        print(f'Info: Failed to overlay image in {max_iter} tries. '
              f'The bottom of an overlay image might be too transparent')
        return np.zeros((window_height, window_width, 4)) if never_return_none else None

    if window_min is None or window_max is None:
        s = '' if window_min is None else 'in'
        print(
            f'Info: Failed to overlay image in {max_iter} tries. An overlay image might be '
            f'too {s}transparent')
        # return whatever the last window was
        return img[y:y + window_height, x:x + window_width, :] if never_return_none else None

    # return the window with the ratio of nontransparent pixels closest to the requirement
    return window_min if min_nontransparent_ratio - max_so_far < \
                         min_so_far - max_nontransparent_ratio else window_max


class OverlayOperation(ImageProcessingStep):
    """Overlays a random image from a folder onto a background image, e.g. branches,
    stickers, parts of cars/trucks. Applies randomized transformations to the overlaid
    image."""

    def __init__(
            self,
            overlay_images_dir: str,
            # Specify if rotation is allowed
            rotate: bool = True,
            # Bigger values lead to a bigger part of the road sign covered by obstacles,
            # hence harder for the SUT
            min_nontransparent_ratio: float = 0.0,
            max_nontransparent_ratio: float = 1.0,
            # Set to >0 to ensure are no sharp cuts at the bottom of the image
            min_nontransparent_ratio_bottom: float = 0.0,
            # randomized parameters, will be initialized randomly if not passed to the
            # constructor
            overlay_image_number: Optional[int] = None,
            rotation_degrees: Optional[int] = None,
            scaling_factor: Optional[Union[float, Tuple[float, float]]] = (0.5, 1.2),
            # from superclass
            name: Optional[str] = None
    ):
        """
        Find a random overlay image in the specified directory. Apply randomized
        transformations to it. Recommended: if overlaying images which are sensitive to
        rotation (e.g. buildings, vehicles), set rotate=False,
        min_nontransparent_ratio_bottom=0.7

        @param overlay_images_dir: Directory containing PNG images which will be overlaid
        onto the road sign

        @param rotate: True if rotation is allowed

        @param min_nontransparent_ratio: Minimum ratio of the original image covered by
        nontransparent pixels of overlaid image

        @param max_nontransparent_ratio: Maximum ratio of the original image covered by
        nontransparent pixels of overlaid image

        @param min_nontransparent_ratio_bottom: Maximum ratio of nontransparent pixels in
        the bottom row of the overlay image. Used to prevent e.g. bottom parts of car images
        from being overlaid on road signs

        @param overlay_image_number: Randomized parameter. Number of the image in
        overlay_images_dir

        @param rotation_degrees: Randomized parameter. 0-360 degrees clockwise rotation

        @param scaling_factor: Randomized parameter. Scales the image before overlaying it.
        Pass an int value to scale the overlay image this many times. Pass a tuple to
        randomly initialize scaling_factor between the min and max values of the tuple.

        @param name: Name of the visual processing step. See superclass
        """
        if name is None:
            name = f'overlay_{overlay_images_dir.split(os.sep)[-1]}_covering_' \
                   f'{min_nontransparent_ratio}_to_' \
                   f'{max_nontransparent_ratio}_of_original_image'
        super().__init__(name)

        # self.overlay_images_dir = overlay_images_dir
        # self.overlay_no_rotation_dir = overlay_no_rotation_dir

        self._min_nontransparent_ratio = min_nontransparent_ratio
        self._max_nontransparent_ratio = max_nontransparent_ratio
        self._min_nontransparent_ratio_bottom = min_nontransparent_ratio_bottom
        self._rotation_allowed = rotate

        self._overlay_image_paths = []
        for file in glob.glob(os.path.join(overlay_images_dir, "*.png")):
            self._overlay_image_paths.append(file)

        # if list of paths is empty
        if not self._overlay_image_paths:
            print(f'Info: no images found in {overlay_images_dir}')
            self._overlay_image_number = -1
        elif overlay_image_number is None:
            self._overlay_image_number = random.randrange(len(self._overlay_image_paths))
        else:
            self._overlay_image_number = overlay_image_number

        if type(scaling_factor) is tuple:
            self._scaling_factor = random.uniform(min(scaling_factor), max(scaling_factor))
        else:
            self._scaling_factor = scaling_factor
        self._rotation_degrees = rotation_degrees if rotation_degrees is not None \
            else random.randrange(0, 360)

    def apply(self, fn_input: RGBImage) -> Optional[RGBImage]:
        """ Overlay the overlay image on fn_input. """
        background_img: np.ndarray = fn_input.value
        background_img_copy = background_img.copy()

        # if no images found in self._overlay_images_dir
        if not self._overlay_image_paths or self._overlay_image_number == -1:
            return None

        index = self._overlay_image_number % len(self._overlay_image_paths)
        foreground_path = self._overlay_image_paths[index]
        if not os.path.exists(foreground_path):
            raise FileNotFoundError(f'File {foreground_path} not found')

        rotate_operation = RotateAndRescale(
            angle_in_degrees=self._rotation_degrees if self._rotation_allowed else 0,
            scale=self._scaling_factor
        )
        print(foreground_path)
        rotated_fg = rotate_operation.apply(RGBAImage(foreground_path))

        window = _find_window(rotated_fg,
                              window_width=background_img.shape[1],
                              window_height=background_img.shape[0],
                              min_nontransparent_ratio=self._min_nontransparent_ratio,
                              max_nontransparent_ratio=self._max_nontransparent_ratio,
                              min_nontransparent_ratio_bottom=
                              self._min_nontransparent_ratio_bottom,
                              never_return_none=False
                              )
        if window is None:
            print('Info: Failed to find a suitable window to cut from the overlaid image')
            return None

        # # Calculate ratio of nontransparent pixels in the best window alphas = window[:,
        # :, 3] nontransparent_ratio = np.count_nonzero(alphas) / (
        # rgb_image.RGBImage.IMAGE_SIZE ** 2) if nontransparent_ratio <
        # self._min_nontransparent_ratio \ or nontransparent_ratio >
        # self._max_nontransparent_ratio: print('Info: Window which is to be overlaid
        # doesn't have a suitable nontransparent ratio') return None

        result_img = Overlay(foreground=RGBAImage(window)).apply(RGBImage(background_img))
        nontransparent_ratio = np.sum(result_img.value != background_img_copy) / (
                3 * RGBImage(background_img).value.shape[0] *
                RGBImage(background_img).value.shape[1])
        if (self._min_nontransparent_ratio < nontransparent_ratio <
                self._max_nontransparent_ratio):
            return result_img
        print('Info: Nontransparent ratio of the window out of boundaries')
        return None


if __name__ == '__main__':
    ''' Demonstrate overlaying of different images on a 'Forbidden' sign '''
    # OVERLAY_IMAGES_DIR = 'examples/gtsrb/assets/overlay/buildings'
    background = RGBImage.read_from_disk(
        'examples/gtsrb/assets/forbidden.png')

    fails = 0
    runs = 0
    for i in range(10):
        print(i)
        op = OverlayOperation(
            overlay_images_dir="examples/gtsrb/assets/overlay/poles",
            # between 5% and 70% of road sign covered with overlaid image
            min_nontransparent_ratio=0.05,
            max_nontransparent_ratio=0.7,
            # ensures poles are not rotated or lifted off the ground
            rotate=False,
            min_nontransparent_ratio_bottom=0.001,
            # randomized variables
            scaling_factor=2.2,
        )
        overlay_img = op.apply(background)
        if overlay_img is None:
            fails += 1
        else:
            overlay_img.plot()
        runs += 1
    print(fails, 'fails out of', runs)
