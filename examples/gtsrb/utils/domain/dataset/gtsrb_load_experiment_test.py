import glob
import json
from typing import List, Optional, Tuple

import PIL
import PIL.Image

from examples.gtsrb.utils.domain.dataset import load_image_strategy


class GTSRBLoadExperimentTestDefault(load_image_strategy.LoadImageStrategy):

    def __init__(
            self,
            experiment_test_fold_glob: str,
            json_tree_path: List[Tuple[str, Optional[int]]]
    ):
        """
        Strategy constructor.

        Arguments:
            experiment_test_fold_glob:
                Location of the test fold that override the normal GTSRB_Images test fold.
            json_tree_path: It is assumed that the PNG image has the same name as
                the JSON metadata file. Parameter describes the tree path one needs to
                take in order to access the class of an image. The path must be presented
                as a list of tuples of format (K, None) or (K, [integer]). A value of None
                indicates recursing into the value found at key K. An integer value of k
                indicated that the value associated with K is a list and recursion
                should happen into the nested dictionary at index K.

        """
        self._json_tree_path = json_tree_path
        self._experiment_test_filepaths: List[Tuple[str, int]] = []

        experiment_filepaths = glob.glob(experiment_test_fold_glob)
        if experiment_filepaths is None or len(experiment_filepaths) == 0:
            raise ValueError(
                "Could not find target experiment dataset in 'assets' folder"
            )

        for path in experiment_filepaths:
            file_name, file_ext = path.rsplit('.', 1)[0], path.rsplit('.', 1)[1]
            assert (
                file_ext == 'png', "Only png files should be loaded, check the [mt_test_glob]"
            )
            # Now read the associated JSON to find the image class
            img_cls = \
                self.__read_cls_label_from_json(f"{file_name}.json")
            # Override the attribute to pack
            self._experiment_test_filepaths.append((path, img_cls))

    def __read_cls_label_from_json(self, path_to_json: str) -> int:
        """Use the metadata from the MT and fuzzy frameworks to identify labels of image"""
        with open(path_to_json) as fp:
            json_tree = json.load(fp)
            for key_name, idx in self._json_tree_path:
                json_tree = json_tree[key_name]
                if idx is not None:
                    # If it's not None, the property
                    # is not nested in an array
                    json_tree = json_tree[idx]
            # By end of traversal we should have locked in
            # the class label, an int
            assert isinstance(json_tree, int)
        return json_tree

    def load_next_image(self) -> Optional[Tuple["PIL.Image", int]]:
        for image_path, img_cls in self._experiment_test_filepaths:
            img = PIL.Image.open(image_path).convert("RGB")
            yield img, img_cls
