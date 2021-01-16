"""
Real Solid Harmonics and Spherical Harmonics
The evalulation scheme and ordering of representations follows:
 Helgaker, Trygve, Poul Jorgensen, and Jeppe Olsen. Molecular electronic-structure theory.
 John Wiley & Sons, 2014. Page 215, Eq. 6.4.47
"""

import torch
from scipy.special import binom, factorial
from torch_gauge.o3.spherical import SphericalTensor


def vm(m):
    return (1 / 2) * (m < 0).long()


def get_c_lmtuv(l, m, t, u, v):

    c = (
            (-1) ** (t + v - vm(m))
            * (1 / 4) ** t
            * binom(l, t)
            * binom(l - t, torch.abs(m) + t)
            * binom(t, u)
            * binom(torch.abs(m), 2 * v)
    )
    assert c != 0
    return c


def get_ns_lm(l, m):
    return 1 / (2 ** torch.abs(m) * factorial(l)) \
           * torch.sqrt(2 * factorial(l + torch.abs(m)) * factorial(l - torch.abs(m)) / 2 ** (m == 0))


def get_xyzcoeff_lm(l, m):
    ts, us, vs = [], [], []
    for t in torch.arange((l - torch.abs(m)) // 2 + 1):
        for u in torch.arange(t):
            for v in torch.arange(vm(m), torch.floor(torch.abs(m)/2 - vm(m)).long() + vm(m) + 1):
                ts.append(t)
                us.append(u)
                vs.append(v)
    ts, us, vs = torch.cat(ts), torch.cat(us), torch.cat(vs)
    xpows_lm = 2 * ts + torch.abs(m) - 2 * (us + vs)
    ypows_lm = 2 * (us + vs)
    zpows_lm = l - 2 * ts - torch.abs(m)
    xyzpows_lm = torch.stack([xpows_lm, ypows_lm, zpows_lm], dim=0)
    clm_tuv = get_c_lmtuv(l, m, ts, us, vs)
    return clm_tuv, xyzpows_lm


class RSHxyz(torch.nn.Module):
    """
    Using pre-generated powers of x,y,z components, stored as a (i_{tuv}^{lm} x 3) tensor
    and the combination coefficients C^{lm}_{tuv}, and a pointer tensor {l(l+1)+m}_i
    Then using torch.prod() and scatter_add to broadcast into real solid harmonics
    """

    def __init__(self, max_l):
        super().__init__()
        self.max_l = max_l
        self._init_coefficients()

    def _init_coefficients(self):
        dst_pointers, xyzpows, ns_lms = [], [], []
        for l in torch.arange(self.max_l):
            for m in torch.arange(-l, l+1):
                ns_lm = get_ns_lm(l, m)
                clm_tuv, xyzpowlm = get_xyzcoeff_lm(l, m)
                dst_pointer = torch.ones_like(clm_tuv) * (l*(l+1)+m)
                dst_pointers.append(dst_pointer)
                xyzpows.append(xyzpowlm)
                ns_lms.append(ns_lm)
        self.dst_pointers = torch.nn.Parameter(torch.cat(dst_pointers), requires_grad=False)
        self.xyzpows = torch.nn.Parameter(torch.cat(xyzpows), requires_grad=False)
        self.ns_lms = torch.nn.Parameter(torch.cat(ns_lms), requires_grad=False)

    def forward(self, xyz) -> SphericalTensor:
        in_shape = xyz.shape
        xyz = xyz.view(-1, 3)
        xyz_poly = torch.pow(xyz.unsqueeze(-1), self.xyzpows.unsqueeze(0)).prod(dim=1)
        out = torch.zeros(xyz.shape[0], self.ns_lms.shape[0], device=xyz.device)
        out = out.scatter_add_(
            dim=1,
            index=self.dst_pointers.unsqueeze(0).expand_as(out),
            src=xyz_poly,
        )
        out = out.view(*in_shape[:-1], self.ns_lms.shape[0])
        return SphericalTensor(
            out,
            rep_dims=(out.dims(),),
            metadata=torch.ones(self.max_l.item(), dtype=torch.long),
        ).to(xyz.device)
