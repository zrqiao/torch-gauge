"""
Geometric operations on point-clouds and manifolds
"""
import math

import torch


class UnivecAngle(torch.autograd.Function):
    """
    Backward-robust angle calculation between two unit vectors
    See https://scicomp.stackexchange.com/questions/27689/numerically-stable-way-of-computing-angles-between-vectors
    Also see https://www.cs.utexas.edu/users/evouga/uploads/4/5/6/8/45689883/turning.pdf which is algebraically
    simpler but involves more forward arithmetic operations
    On the benchmark CPU, tensor.norm() is much slower than dot and 2x slower than
    cross product. We take the second approach here
    """

    @staticmethod
    def forward(ctx, vec1, vec2):
        cross_vec = torch.cross(vec1, vec2, dim=-1)
        cross_norm = cross_vec.pow(2).sum(dim=-1).sqrt()
        vec_dot = vec1.mul(vec2).sum(-1)
        angle = torch.atan2(cross_norm, 1 + vec_dot).mul_(2)
        ctx.save_for_backward(vec1, vec2, cross_vec, cross_norm)
        return angle

    @staticmethod
    def backward(ctx, grad_output):
        vec1, vec2, cross_vec, cross_norm = ctx.saved_tensors
        cross_norm = cross_norm.unsqueeze(-1).expand_as(cross_vec)
        z = cross_vec.div(cross_norm)
        # Need to consider using explicit Chebyshev-poly if this breaks in near-edge cases
        z.masked_fill_(cross_norm == 0, 0)
        grad_vec1, grad_vec2 = torch.cross(vec1, z), torch.cross(vec2, z).neg_()
        return grad_vec1.mul(grad_output.unsqueeze(-1)), grad_vec2.mul(
            grad_output.unsqueeze(-1)
        )


def univec_angle(vec1: torch.Tensor, vec2: torch.Tensor):
    return UnivecAngle.apply(vec1, vec2)


def univec_angle_unsafe(vec1: torch.Tensor, vec2: torch.Tensor):
    return torch.acos(vec1.mul(vec2).sum(-1))


def univec_cos(vec1: torch.Tensor, vec2: torch.Tensor):
    """Returns cos(\theta) for angular embeddings"""
    return vec1.mul(vec2).sum(-1)


def rotation_matrix_xyz(phi, axis, dtype=torch.double):
    """
    Rotation matrix for the intrinsic rotation about the specified axis,
     with the column-xyz (left-multiplication) convention
    """
    if axis == "z":
        return torch.tensor(
            [
                [math.cos(phi), -math.sin(phi), 0],
                [math.sin(phi), math.cos(phi), 0],
                [0, 0, 1],
            ],
            dtype=dtype,
        )
    elif axis == "y":
        return torch.tensor(
            [
                [math.cos(phi), 0, math.sin(phi)],
                [0, 1, 0],
                [-math.sin(phi), 0, math.cos(phi)],
            ],
            dtype=dtype,
        )
    elif axis == "x":
        return torch.tensor(
            [
                [1, 0, 0],
                [0, math.cos(phi), -math.sin(phi)],
                [0, math.sin(phi), math.cos(phi)],
            ],
            dtype=dtype,
        )
    else:
        raise ValueError(f"Invalid axis: {axis}")


class Chebyshev(torch.nn.Module):
    """Generating Chebshev polynomials from powers"""

    def __init__(self, order):
        super(Chebyshev, self).__init__()
        self.order = order
        self.orders = torch.arange(order + 1, dtype=torch.long)
        self.orders = torch.nn.Parameter(self.orders, requires_grad=False)
        self._init_coefficients()

    def _init_coefficients(self):
        """
        Generating combination coefficients using recursion
        Starting from order 1
        """
        coeff = torch.zeros(self.order + 1, self.order + 1)
        coeff[0, 0] = 1
        coeff[1, 1] = 1
        for idx in range(2, self.order + 1):
            coeff[1:, idx] = 2 * coeff[:-1, idx - 1]
            coeff[:, idx] -= coeff[:, idx - 2]
        self.cheb_coeff = torch.nn.Parameter(coeff[:, 1:], requires_grad=False)

    def forward(self, x):
        size = (*x.shape, self.order + 1)
        x = x.unsqueeze(-1).expand(size).pow(self.orders)
        out = torch.matmul(x, self.cheb_coeff)
        return out


def poly_env(d, p=6):
    """A polynomial wrapper defined in J. Klicpera, J. Groß,
    and S. G¨unnemann, arXiv preprint arXiv:2003.03123 (2020)"""
    return (
        1
        - (p + 1) * (p + 2) // 2 * d ** p
        + p * (p + 2) * d ** (p + 1)
        - p * (p + 1) // 2 * d ** (p + 2)
    )
