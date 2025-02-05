import abc
import typing

import numpy as np

from examples.gtsrb.utils.domain.model import model_adapter


class KerasModelAdapter(model_adapter.ModelAdapter, abc.ABC):
    """Parent class for all models developed in the Keras framework.

    This is an incomplete implementation of the ModelAdapter interface.
    The [get_latent_representation] method is expected to be implemented
    on a model by model case.
    """

    def weights(self) -> typing.Sequence[np.ndarray]:
        return [np.array(layer.weights) for layer in self.model.layers]
