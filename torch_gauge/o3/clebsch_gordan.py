"""
SO(3) Clebsch-Gordan coefficients and the coupler
"""


import torch

from torch_gauge.o3.spherical import SphericalTensor


class CeviLevitaCoupler(torch.nn.Module):
    """
    Simple tensor coupling module when max_l==1.
    """

    def __init__(self, metadata: torch.LongTensor):
        super().__init__()
        assert metadata.dim() == 1
        assert (
            len(metadata) == 2
        ), "Only SphericalTensor of max degree 1 is applicable for Cevi-Levita"
        self._metadata = metadata

    def forward(self, x1: SphericalTensor, x2: SphericalTensor):
        assert x1.metadata.shape[0] == 1
        assert x2.metadata.shape[0] == 1
        assert torch.all(x1.metadata[0].eq(self._metadata))
        assert torch.all(x2.metadata[0].eq(self._metadata))
        raise NotImplementedError
