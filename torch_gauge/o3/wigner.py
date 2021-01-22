"""
Wigner (rotation) matrices for complex and real spherical harmonics
Evaluation and symbolic convention follows:
 Sakurai, J. J. "Modern Quantum Mechanics 2Nd Edition." Person New International edition (2014).
 Page 238, Eq. 3.9.33.
"""

import math

from joblib import Memory

memory = Memory(".o3_cache", verbose=0)
import torch
from scipy.special import factorial


@memory.cache()
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
    return sgn * numer * denom


def wigner_small_d_csh(j, beta):
    """Rotating complex spherical harmonics by the y axis"""
    small_d = torch.zeros(2 * j + 1, 2 * j + 1, dtype=torch.double)
    for m in range(-j, j + 1):
        for m1 in range(-j, j + 1):
            kmin = min(0, m - m1)
            kmax = max(0, j + m, j - m1)
            d_m1_m = 0
            for k in range(kmin, kmax + 1):
                angular = (math.cos(beta / 2)) ** (2 * j - 2 * k + m - m1) * (
                    math.sin(beta / 2)
                ) ** (2 * j - m + m1)
                d_m1_m += _wigner_small_d_coeff(j, m1, m, k) * angular
            small_d[m1+j, m+j] = d_m1_m
    return small_d


def wigner_D_csh(j, alpha, beta, gamma):
    """Euler rotation for complex spherical harmonics"""
    ms = torch.arange(-j, j+1)
    small_d = wigner_small_d_csh(j, beta).to(torch.cdouble)
    alpha_phases = torch.exp(-1j * ms * alpha)
    gamma_phases = torch.exp(-1j * ms * gamma)
    wigner_D = alpha_phases.unsqueeze(1) * small_d * gamma_phases.unsqueeze(0)
    return wigner_D
