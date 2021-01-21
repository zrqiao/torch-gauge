"""
Geometric operations on point-clouds and manifolds
"""
import torch
import torch.nn as nn


class UnivecAngle(torch.autograd.Function):
    """
    Backward-robust angle calculation between two unit vectors
    See https://scicomp.stackexchange.com/questions/27689/numerically-stable-way-of-computing-angles-between-vectors
    Also see https://www.cs.utexas.edu/users/evouga/uploads/4/5/6/8/45689883/turning.pdf which is algebraically
    simpler but involves more forward arithmetic operations
    On the benchmark CPU, tensor.norm() is much slower than dot and 2x slower than
    cross product. We take the second approach
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
        # We need to consider using explicit Chebyshev-poly if this breaks in near-edge cases
        z.masked_fill_(cross_norm == 0, 0)
        grad_vec1, grad_vec2 = torch.cross(vec1, z), torch.cross(vec2, z).neg_()
        return grad_vec1.mul(grad_output.unsqueeze(-1)), grad_vec2.mul(grad_output.unsqueeze(-1))


def univec_angle(vec1: torch.Tensor, vec2: torch.Tensor):
    return UnivecAngle.apply(vec1, vec2)


def univec_angle_unsafe(vec1: torch.Tensor, vec2: torch.Tensor):
    return torch.acos(vec1.mul(vec2).sum(-1))


def univec_cos(vec1: torch.Tensor, vec2: torch.Tensor):
    """Returns cos(\theta) for angular embeddings"""
    return vec1.mul(vec2).sum(-1)
