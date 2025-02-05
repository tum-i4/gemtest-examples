import glob
import re
from typing import List, Optional, Tuple, Literal

import PIL.Image

from examples.gtsrb.utils.domain.dataset import load_image_strategy


class GTSRBLoadTrainDefault(load_image_strategy.LoadImageStrategy):

    def __init__(
            self,
            seeked_classes: Optional[List[int]],
            gtsrb_training_glob: Optional[str],
            image_mode: Literal['RGB', 'RGBA'] = 'RGB'
    ):
        if seeked_classes is None:
            self._seeked_classes = list(range(43))
        else:
            self._seeked_classes = seeked_classes
        self._image_mode = image_mode
        train_filepaths = glob.glob(gtsrb_training_glob)
        self._train_filepaths = train_filepaths
        if train_filepaths is None or len(train_filepaths) == 0:
            raise ValueError("Could not find GTSRB_Images dataset in 'assets' folder")

    def _load_train_image(
            self, image_path: str
    ) -> Optional[Tuple["PIL.Image", int]]:
        if not re.fullmatch(r"^.*\.png$", image_path):
            return None
        image_class = int(image_path.split('/')[-2])
        if image_class not in self._seeked_classes:
            return None
        return_img = PIL.Image.open(image_path).convert(self._image_mode)
        return return_img, image_class

    def load_next_image(self) -> Optional[Tuple["PIL.Image", int]]:
        for image_path in self._train_filepaths:
            img_label_tuple = self._load_train_image(image_path)
            if img_label_tuple is None:
                continue
            yield img_label_tuple
