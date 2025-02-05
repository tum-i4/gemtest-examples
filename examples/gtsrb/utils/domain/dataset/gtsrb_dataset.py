from examples.gtsrb.utils.domain import types
from examples.gtsrb.utils.domain.dataset import load_image_strategy, dataset


class GTSRBDataset(dataset.Dataset):
    """Loads GTSRB dataset from local disk one image at a time."""

    def __init__(
            self,
            train_load_strategy: load_image_strategy.LoadImageStrategy,
            test_load_strategy: load_image_strategy.LoadImageStrategy,
    ):
        """
        GTSRB test loader constructor.

        Arguments:
            train_load_strategy: Strategy for loading train images. Only one
                strategy since the train fold is the same between experiments.
                Implemented such for consistency with test_load_strategy.
            test_load_strategy: Strategy for loading the test images. Using strategy
                pattern to allow loading flexibly from either the default fold
                described in CSV or from JSON metadata files.
        """
        self._train_load_strategy = train_load_strategy
        self._test_load_strategy = test_load_strategy

    def train_iterator(self) -> types.DataIterator:
        """Iterates over the train dataset, loading one image from disk.

        Yields:
            Tuple of numpy array of image in RGB format and image class.
        """
        for img_cls_tuple in self._train_load_strategy.load_next_image():
            yield img_cls_tuple

    def test_iterator(self) -> types.DataIterator:
        """Iterates over the test dataset, loading one image from disk.

        Yields:
            Tuple of numpy array of image in RGB format and image class.
        """
        for img_cls_tuple in self._test_load_strategy.load_next_image():
            yield img_cls_tuple

    TOTAL_NUM_CLASSES = 43

    RED_BORDER_CLASSES = [
        0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    ]

    BLUE_BORDER_CLASSES = [33, 34, 35, 36, 37, 38, 39, 40]
