"""
SO(3) Clebsch-Gordan coefficients and the coupler
See Page 225, Eq. 3.8.49 of J. J. Sakurai, and Eq. 6, 9 of
 Schulten, Klaus, and Roy G. Gordon, Journal of Mathematical Physics 16.10 (1975): 1961-1970
"""


import os
import torch

from sympy.physics.quantum.cg import CG
from sympy import N
from torch_gauge.o3.spherical import SphericalTensor
from torch_gauge import ROOT_DIR
from joblib import Memory

memory = Memory(os.path.join(ROOT_DIR, ".o3_cache"), verbose=0)


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


@memory.cache
def get_clebsch_gordan_coefficient(j1, j2, j, m1, m2, m):
    """
    Generate Clebsch-Gordan coefficients using sympy with caching
    """
    return N(CG(j1, m1, j2, m2, j, m).doit())

