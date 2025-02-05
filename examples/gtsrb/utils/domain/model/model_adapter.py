import abc
import typing

import numpy as np


class ModelAdapter(abc.ABC):
    """Adapter class for the various deep learning models.

    The domain problem of this framework demands that this interface
    is to be used on a model by model basis.
    """

    def __init__(self, model: typing.Any):
        self._model = model

    @property
    def model(self) -> typing.Any:
        """Returns underlying model."""
        return self._model

    def weights(self) -> typing.Sequence[np.ndarray]:
        """Returns an iterable of all trainable weights in the model.

        Interface dependants expect np.ndarray neutral format.
        """
        raise NotImplementedError

    def get_latent_representation(self, one_input) -> np.ndarray:
        """Get the latent representation of one input tensor in the model space.

        An example of latent space are the features outputted by convolutional
        layers of a CNN before the classification head.

        Models in the framework's domain such as Centroid Positioning make assertions
        of the quality of the latent space outputted by a model.
        """
        raise NotImplementedError
