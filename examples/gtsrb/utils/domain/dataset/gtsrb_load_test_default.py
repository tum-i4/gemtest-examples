import glob
import re
from typing import List, Optional, Tuple, Literal

import PIL.Image
import pandas as pd

from examples.gtsrb.utils.domain.dataset import load_image_strategy


class GTSRBLoadTestDefault(load_image_strategy.LoadImageStrategy):

    def __init__(
            self,
            seeked_classes: Optional[List[int]],
            gtsrb_testing_glob: Optional[str],
            gtsrb_testing_ground_truth_path: str,
            image_mode: Literal['RGB', 'RGBA'] = 'RGB'
    ):
        if seeked_classes is None:
            self._seeked_classes = list(range(43))
        else:
            self._seeked_classes = seeked_classes
        self._image_mode = image_mode
        test_filepaths = glob.glob(gtsrb_testing_glob)
        if test_filepaths is None or len(test_filepaths) == 0:
            raise ValueError("Could not find GTSRB_Images dataset in 'assets' folder")
        self._test_filepaths = test_filepaths
        self._test_ground_truth = {}
        self._load_labels_test_dataset(gtsrb_testing_ground_truth_path)

    def _load_labels_test_dataset(self, gt_path: str):
        """Load the ground-truth labels for test dataset from CSV descriptor file."""
        test_ground_truth_series = pd.read_csv(
            gt_path, delimiter=','
        )
        for _, row in test_ground_truth_series.iterrows():
            fn = row['Path'].split('/')[1]
            self._test_ground_truth[fn] = \
                int(row['ClassId'])

    def _load_test_image(
            self, image_path: str
    ) -> Optional[Tuple["PIL.Image", int]]:
        if not re.fullmatch(r"^.*\.png$", image_path):
            return None
        fn = image_path.split('/')[-1]
        image_class = self._test_ground_truth[fn]
        if image_class not in self._seeked_classes:
            return None
        return_img = PIL.Image.open(image_path).convert(self._image_mode)
        return return_img, image_class

    def load_next_image(self) -> Optional[Tuple["PIL.Image", int]]:
        for image_path in self._test_filepaths:
            img_label_tuple = self._load_test_image(image_path)
            if img_label_tuple is None:
                continue
            yield img_label_tuple
