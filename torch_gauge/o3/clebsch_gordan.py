"""
SO(3) Clebsch-Gordan coefficients and the coupler
See Page 225, Eq. 3.8.49 of J. J. Sakurai, and Eq. 6, 9 of
 Schulten, Klaus, and Roy G. Gordon, Journal of Mathematical Physics 16.10 (1975): 1961-1970
"""


import os

import torch
from joblib import Memory
from sympy import N
from sympy.physics.quantum.cg import CG

from torch_gauge import ROOT_DIR
from torch_gauge.o3.spherical import SphericalTensor

memory = Memory(os.path.join(ROOT_DIR, ".o3_cache"), verbose=0)


class LeviCivitaCoupler(torch.nn.Module):
    """
    Simple tensor coupling module when max_l==1.
    The input spherical tensors must have n_rep_dims==1 and aligned dimensions
    """

    def __init__(self, metadata: torch.LongTensor):
        super().__init__()
        assert metadata.dim() == 1
        assert (
            len(metadata) == 2
        ), "Only SphericalTensor of max degree 1 is applicable for Cevi-Levita"
        self._metadata = metadata

    def forward(self, x1: SphericalTensor, x2: SphericalTensor, overlap_out=True):
        assert x1.metadata.shape[0] == 1
        assert x2.metadata.shape[0] == 1
        assert x1.rep_dims[0] == x2.rep_dims[0]
        coupling_dim = x1.rep_dims[0]
        assert torch.all(x1.metadata[0].eq(self._metadata))
        assert torch.all(x2.metadata[0].eq(self._metadata))
        ten_l1_1 = x1.ten.narrow(
            coupling_dim, self._metadata[0], self._metadata[1] * 3
        ).unflatten(coupling_dim, (3, self._metadata[1]))
        ten_l1_2 = x2.ten.narrow(
            coupling_dim, self._metadata[0], self._metadata[1] * 3
        ).unflatten(coupling_dim, (3, self._metadata[1]))
        # 0x0->0
        out_000 = x1.ten.narrow(coupling_dim, 0, self._metadata[0]) * x2.ten.narrow(
            coupling_dim, 0, self._metadata[0]
        )
        # 0x1->1
        out_011 = (
            x1.ten.narrow(coupling_dim, 0, self._metadata[1])
            .unsqueeze(coupling_dim)
            .mul(ten_l1_2)
        )
        # 1x0->1
        out_101 = (
            x2.ten.narrow(coupling_dim, 0, self._metadata[1])
            .unsqueeze(coupling_dim)
            .mul(ten_l1_1)
        )
        # 1x1->0
        out_110 = (ten_l1_1 * ten_l1_2).sum(coupling_dim)
        # 1x1->1, note that cross works since (y,z,x) is a canonical order
        out_111 = torch.cross(ten_l1_1, ten_l1_2, dim=coupling_dim)
        if overlap_out:
            # Align and contract the coupling outputs
            out_l0 = out_000
            out_l0.narrow(coupling_dim, 0, self._metadata[1]).add(out_110)
            out_l1 = (
                out_111.add(out_101)
                .add(out_011)
                .flatten(coupling_dim, coupling_dim + 1)
            )
            return x1.self_like(torch.cat([out_l0, out_l1], dim=coupling_dim))
        else:
            # Concatenate the coupling outputs to form a augmented tensor
            out_l0 = torch.cat([out_000, out_110], dim=coupling_dim)
            out_l1 = torch.cat(
                [out_101, out_011, out_111], dim=coupling_dim + 1
            ).flatten(coupling_dim, coupling_dim + 1)
            return SphericalTensor(
                torch.cat([out_l0, out_l1], dim=coupling_dim),
                rep_dims=(coupling_dim,),
                metadata=torch.LongTensor(
                    [[self._metadata[0] + self._metadata[1], self._metadata[1] * 3]]
                ),
            )


@memory.cache
def get_clebsch_gordan_coefficient(j1, j2, j, m1, m2, m):
    """
    Generate Clebsch-Gordan coefficients using sympy with caching
    """
    return N(CG(j1, m1, j2, m2, j, m).doit())
