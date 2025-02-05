import abc
import typing

import numpy as np
import torch

from examples.gtsrb.utils.domain.model import model_adapter


class TorchModelAdapter(model_adapter.ModelAdapter, abc.ABC):
    """Parent class for all models developed in the Pytorch framework.

    This is an incomplete implementation of the ModelAdapter interface.
    The [get_latent_representation] method is expected to be implemented
    on a model by model case.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)

    def weights(self) -> typing.Sequence[np.ndarray]:
        """Architecture agnostic implementation for obtaining the weights from PyTorch model.

        TODO: Needs testing
        """
        model: torch.nn.Module = self.model
        return [param.data.numpy().flatten() for param in model.parameters()]
