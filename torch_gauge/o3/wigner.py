"""
Wigner (rotation) matrices for complex and real spherical harmonics
Evaluation and symbolic convention follows:
 Sakurai, J. J. "Modern Quantum Mechanics 2Nd Edition." Person New International edition (2014).
 Page 238, Eq. 3.9.33.
"""

import math
import os

import torch
from joblib import Memory
from scipy.special import factorial

from torch_gauge import ROOT_DIR

memory = Memory(os.path.join(ROOT_DIR, ".o3_cache"), verbose=0)


@memory.cache
def _wigner_small_d_coeff(j, m1, m, k):
    sgn = (-1) ** (k - m + m1)
    numer = math.sqrt(
        factorial(j + m) * factorial(j - m) * factorial(j + m1) * factorial(j - m1)
    )
    denom = (
        factorial(j + m - k)
        * factorial(k)
        * factorial(j - k - m1)
        * factorial(k - m + m1)
    )
    if denom == 0:
        return None
    else:
        return sgn * numer / denom


def wigner_small_d_csh(j, beta):
    """Rotating complex spherical harmonics by the y axis"""
    small_d = torch.zeros(2 * j + 1, 2 * j + 1, dtype=torch.double)
    for m in range(-j, j + 1):
        for m1 in range(-j, j + 1):
            kmin = min(0, m - m1)
            kmax = max(0, j + m, j - m1)
            d_m1_m = 0
            for k in range(kmin, kmax + 1):
                prefactor = _wigner_small_d_coeff(j, m1, m, k)
                if prefactor is not None:
                    angular = (math.cos(beta / 2)) ** (2 * j - 2 * k + m - m1) * (
                        math.sin(beta / 2)
                    ) ** (2 * k - m + m1)
                    d_m1_m += prefactor * angular
            small_d[m1 + j, m + j] = d_m1_m
    return small_d


def wigner_D_csh(j, alpha, beta, gamma):
    """
    Euler rotation for complex spherical harmonics
    Contra-variant convention, for right multiplication
    """
    ms = torch.arange(-j, j + 1)
    small_d = wigner_small_d_csh(j, beta).to(torch.cdouble)
    alpha_phases = torch.exp(-1j * ms * alpha)
    gamma_phases = torch.exp(-1j * ms * gamma)
    wigner_D = alpha_phases.unsqueeze(1) * small_d * gamma_phases.unsqueeze(0)
    assert torch.allclose(
        wigner_D.mm(wigner_D.t().conj()),
        torch.eye(2 * j + 1, 2 * j + 1, dtype=torch.cdouble),
        atol=1e-7,
    )
    return wigner_D


@memory.cache
def csh_to_rsh(j):
    transform_mat = torch.zeros(2 * j + 1, 2 * j + 1, dtype=torch.cdouble)
    for m in range(-j, j + 1):
        # Enumerate by out m
        if m < 0:
            transform_mat[j + m, j + m] = 1j / math.sqrt(2)
            transform_mat[j - m, j + m] = -((-1) ** m) * 1j / math.sqrt(2)
        elif m == 0:
            transform_mat[j + m, j + m] = 1
        elif m > 0:
            transform_mat[j - m, j + m] = 1 / math.sqrt(2)
            transform_mat[j + m, j + m] = (-1) ** m / math.sqrt(2)
        else:
            raise ValueError
    return transform_mat


def wigner_D_rsh(j, alpha, beta, gamma):
    c2r = csh_to_rsh(j)
    wigner_csh = wigner_D_csh(j, alpha, beta, gamma)
    wigner_rsh = (c2r.conj().t()).mm(wigner_csh.t().conj()).mm(c2r)
    # Checking the RSH rotation matrix entries are all real
    assert torch.allclose(
        wigner_rsh.imag, torch.zeros(2 * j + 1, 2 * j + 1, dtype=torch.double)
    ), print(c2r, wigner_csh, wigner_rsh)
    return wigner_rsh.real
