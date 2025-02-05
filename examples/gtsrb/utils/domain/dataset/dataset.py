import abc
from typing import Tuple, Generator, Union

import numpy as np

from examples.gtsrb.utils.domain.image import rgb_image


class Dataset(abc.ABC):
    """Dataset interface describing the required functionalities for
    calculating latent over it. See metamorphic_test.latent.dataset
    module for usages.

    Every particular dataset used in testing is expected to have a
    companion object instantiating this interface.
    """

    def train_iterator(
            self
    ) -> Generator[Tuple[Union[rgb_image.RGBImage, np.ndarray], int], None, None]:
        """Yields (tensor, class_label) tuples.

        None should be returned if the dataset cannot be iterated over e.g.
        due to bad init arguments.
        """
        raise NotImplementedError

    def test_iterator(
            self
    ) -> Generator[Tuple[Union[rgb_image.RGBImage, np.ndarray], int], None, None]:
        """Yields (tensor, class_label) tuples.

        None should be returned if the dataset cannot be iterated over e.g.
        due to bad init arguments.
        """
        raise NotImplementedError
