"""
Real Solid Harmonics and Spherical Harmonics
The evalulation scheme and ordering of representations follows:
 Helgaker, Trygve, Poul Jorgensen, and Jeppe Olsen. Molecular electronic-structure theory.
 John Wiley & Sons, 2014. Page 215, Eq. 6.4.47
"""

import torch
from scipy.special import binom


def c_lmtuv(l, m, t, u, v):
    def vm(m):
        return (1 / 2) * int(m < 0)

    c = (
        (-1) ** (t + v - vm(m))
        * (1 / 4) ** t
        * binom(l, t)
        * binom(l - t, abs(m) + t)
        * binom(t, u)
        * binom(abs(m), 2 * v)
    )
    assert c != 0
    return c


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
        raise NotImplementedError
