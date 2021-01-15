"""
Real Solid Harmonics and Spherical Harmonics
The evalulation scheme and ordering of representations follows:
 Helgaker, Trygve, Poul Jorgensen, and Jeppe Olsen. Molecular electronic-structure theory.
 John Wiley & Sons, 2014. Page 215, Eq. 6.4.47
"""

# Notes to myself
# Using Pre-generate the required powers of array xyzpow: (N_{tuv}^{lm} x 3)
# and the combination coefficients C^{lm}_{tuv}
# Then using torch.prod() and scatter_add to broadcast into solid harmonics
