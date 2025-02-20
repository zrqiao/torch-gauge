"""
Real Solid Harmonics and Spherical Harmonics (Ket state).

The evalulation scheme, convention and ordering of representations follows:
    Helgaker, Trygve, Poul Jorgensen, and Jeppe Olsen. Molecular electronic-structure theory.
    John Wiley & Sons, 2014. Page 215, Eq. 6.4.47
"""

import os

import torch
from joblib import Memory
from scipy.special import binom, factorial
from torch_gauge import ROOT_DIR
from torch_gauge.o3.spherical import SphericalTensor

memory = Memory(os.path.join(ROOT_DIR, ".o3_cache"), verbose=0)

def torch_factorial(x):
  if x.dim() == 0:
    x = x.unsqueeze(-1)
    out = factorial(x)
    return torch.from_numpy(out).squeeze()
  out = factorial(x)
  return torch.from_numpy(out)

def vm(m):
    return (1 / 2) * (m < 0).long()


@memory.cache
def get_c_lmtuv(l, m, t, u, v):

    c = (
        (-1) ** (t + v - vm(m))
        * (1 / 4) ** t
        * binom(l, t)
        * binom(l - t, torch.abs(m) + t)
        * binom(t, u)
        * binom(torch.abs(m), 2 * v)
    )
    assert (c != 0).any()
    return c


@memory.cache
def get_ns_lm(l, m):
    return (1 / (2 ** torch.abs(m) * torch_factorial(l))) * torch.sqrt(
        2
        * torch_factorial(l + torch.abs(m))
        * torch_factorial(l - torch.abs(m))
        / (2 ** (m == 0).long())
    )


def get_xyzcoeff_lm(l, m):
    ts, us, vs = [], [], []
    for t in torch.arange((l - torch.abs(m)) // 2 + 1):
        for u in torch.arange(t + 1):
            for v in torch.arange(
                vm(m), torch.floor(torch.abs(m) / 2 - vm(m)).long() + vm(m) + 1
            ):
                ts.append(t)
                us.append(u)
                vs.append(v)
    ts, us, vs = torch.stack(ts), torch.stack(us), torch.stack(vs)
    xpows_lm = 2 * ts + torch.abs(m) - 2 * (us + vs)
    ypows_lm = 2 * (us + vs)
    zpows_lm = l - 2 * ts - torch.abs(m)
    xyzpows_lm = torch.stack([xpows_lm, ypows_lm, zpows_lm], dim=0)
    clm_tuv = get_c_lmtuv(l, m, ts, us, vs)
    return clm_tuv, xyzpows_lm


class RSHxyz(torch.nn.Module):
    """
    The module to generate Real Spherical Harmonics up to the given order from
    (batched) xyz coordinates.

    Using pre-generated powers of x,y,z components, stored as a (i_{tuv}^{lm} x 3) tensor
    and the combination coefficients C^{lm}_{tuv}, and a pointer tensor {l(l+1)+m}_i
    to track the real spherical harmonics coefficients. Then using
    torch.prod() and scatter_add to broadcast into RSHs.

    Args:
        max_l (int): The maximum order (l) of the desired spherical harmonics.
    """

    def __init__(self, max_l: int):
        super().__init__()
        self.max_l = max_l
        self._init_coefficients()

    def _init_coefficients(self):
        dst_pointers, xyzpows, ns_lms, clmtuvs = [], [], [], []
        for l in torch.arange(self.max_l + 1, dtype=torch.long):
            for m in torch.arange(-l, l + 1, dtype=torch.long):
                ns_lm = get_ns_lm(l, m)
                clm_tuv, xyzpowlm = get_xyzcoeff_lm(l, m)
                dst_pointer = torch.ones_like(clm_tuv) * (l * (l + 1) + m)
                dst_pointers.append(dst_pointer)
                xyzpows.append(xyzpowlm)
                ns_lms.append(ns_lm)
                clmtuvs.append(clm_tuv)
        self.register_buffer("dst_pointers", torch.cat(dst_pointers).long())
        self.register_buffer("clm_tuvs", torch.cat(clmtuvs, dim=0))
        self.register_buffer("xyzpows", torch.cat(xyzpows, dim=1).long())
        self.register_buffer("ns_lms", torch.stack(ns_lms, dim=0))
        self.out_metadata = torch.ones((1, self.max_l + 1), dtype=torch.long)
        self.register_buffer(
            "out_replayout",
            SphericalTensor.generate_rep_layout_1d_(self.out_metadata[0]),
        )

    def forward(self, xyz) -> SphericalTensor:
        """"""
        in_shape = xyz.shape
        xyz = xyz.view(-1, 3)
        # Enforce 0**0=1 for second-order backward stability
        mask = self.xyzpows == 0
        xyz_poly = torch.ones(
            *xyz.shape, self.xyzpows.shape[-1], dtype=xyz.dtype, device=xyz.device
        )
        xyz_poly[:, ~mask] = torch.pow(
            xyz.unsqueeze(-1).expand_as(xyz_poly)[:, ~mask],
            self.xyzpows.unsqueeze(0).expand_as(xyz_poly)[:, ~mask],
        )
        xyz_poly = xyz_poly.prod(dim=1)
        out = torch.zeros(
            xyz.shape[0], self.ns_lms.shape[0], device=xyz.device, dtype=xyz.dtype
        )
        out = out.scatter_add_(
            dim=1,
            index=self.dst_pointers.unsqueeze(0).expand_as(xyz_poly),
            src=xyz_poly * self.clm_tuvs.to(xyz.dtype).unsqueeze(0),
        )
        out = out.mul_(self.ns_lms)
        out = out.view(*in_shape[:-1], self.ns_lms.shape[0])
        # noinspection PyTypeChecker
        return SphericalTensor(
            out,
            rep_dims=(out.dim() - 1,),
            metadata=self.out_metadata,
            rep_layout=(self.out_replayout.data,),
            num_channels=(self.max_l + 1,),
        )
