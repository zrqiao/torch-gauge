"""
Testing Spherical neural operations
"""

import math
import random

import torch

from torch_gauge.nn import IELin, KernelBroadcast, RepNorm1d
from torch_gauge.o3 import SphericalTensor
from torch_gauge.o3.rsh import RSHxyz
from torch_gauge.o3.wigner import wigner_D_rsh


def real_wigner_rot(spten, alpha, beta, gamma):
    # Naive implementation, do not use in models
    rot_ten = torch.zeros_like(spten.ten)
    for l in range(spten.metadata.shape[1]):
        real_wigner_l = wigner_D_rsh(l, alpha, beta, gamma).type(spten.ten.dtype)
        l_mask = spten.rep_layout[0][0] == l
        torot = spten.ten[:, l_mask]
        if torot.numel() == 0:
            continue
        torot = torot.view(spten.ten.shape[0], 2 * l + 1, -1)
        rot_ten[:, l_mask] = (
            torch.einsum("ijk,ja->iak", torot, real_wigner_l)
            .contiguous()
            .view(torot.shape[0], -1)
        )
    return spten.self_like(rot_ten)


def test_IELin_forward_1d():
    metadata = torch.LongTensor([[8, 4, 2, 4, 1]])
    test_sp_ten = SphericalTensor(torch.rand(4, 6, 12, 67), (3,), metadata)
    ielin_dense = IELin(
        torch.LongTensor([8, 4, 2, 4, 1]),
        torch.LongTensor([2, 3, 4, 5, 6]),
        group="so3",
    )
    dense_out = ielin_dense(test_sp_ten)
    assert dense_out.ten.shape == (4, 6, 12, 120)
    assert dense_out.num_channels == (20,)
    assert torch.all(dense_out.metadata.eq(torch.LongTensor([[2, 3, 4, 5, 6]])))

    ielin_dropped = IELin(
        torch.LongTensor([8, 4, 2, 4, 1]),
        torch.LongTensor([2, 0, 1, 3, 0]),
        group="so3",
    )
    dropped_out = ielin_dropped(test_sp_ten)
    assert dropped_out.ten.shape == (4, 6, 12, 28)
    assert dropped_out.num_channels == (6,)
    assert torch.all(dropped_out.metadata.eq(torch.LongTensor([[2, 0, 1, 3, 0]])))


def test_IELin_forward_equivariance():
    alpha = random.random() * 2 * math.pi
    beta = random.random() * 2 * math.pi
    gamma = random.random() * 2 * math.pi
    metadata = torch.LongTensor([[8, 4, 2, 4, 1]])
    test_sp_ten = SphericalTensor(torch.rand(48, 67), (1,), metadata)
    spten_rot = real_wigner_rot(test_sp_ten, alpha, beta, gamma)
    ielin_dense = IELin(
        torch.LongTensor([8, 4, 2, 4, 1]),
        torch.LongTensor([2, 3, 4, 5, 6]),
        group="so3",
    )
    dense_out = ielin_dense(test_sp_ten)
    rot_dense_out = real_wigner_rot(dense_out, alpha, beta, gamma)
    dense_out_rot = ielin_dense(spten_rot)
    assert torch.allclose(
        rot_dense_out.ten, dense_out_rot.ten, atol=1e-6, rtol=1e-6
    ), print((dense_out_rot.ten - rot_dense_out.ten).flatten().abs())

    ielin_dropped = IELin(
        torch.LongTensor([8, 4, 2, 4, 1]),
        torch.LongTensor([2, 0, 1, 3, 0]),
        group="so3",
    )
    dropped_out = ielin_dropped(test_sp_ten)
    dropped_out_rot = ielin_dropped(spten_rot)
    rot_dropped_out = real_wigner_rot(dropped_out, alpha, beta, gamma)
    assert torch.allclose(
        rot_dropped_out.ten, dropped_out_rot.ten, atol=1e-6, rtol=1e-6
    ), print((rot_dropped_out.ten - dropped_out_rot.ten).flatten().abs())


def test_IELin_backward_1d():
    metadata = torch.LongTensor([[8, 4, 2, 4, 1]])
    test_sp_ten = SphericalTensor(torch.rand(4, 6, 12, 67), (3,), metadata)
    test_sp_ten.ten.requires_grad = True
    ielin_dense = IELin(
        torch.LongTensor([8, 4, 2, 4, 1]),
        torch.LongTensor([2, 3, 4, 5, 6]),
        group="so3",
    )
    dense_out = ielin_dense(test_sp_ten)
    pseudo_loss = dense_out.ten.pow(2).sum()
    grad_ten = torch.autograd.grad(outputs=pseudo_loss, inputs=test_sp_ten.ten)
    assert torch.all(grad_ten[0].abs().bool())

    ielin_dropped = IELin(
        torch.LongTensor([8, 4, 2, 4, 1]),
        torch.LongTensor([2, 0, 1, 3, 0]),
        group="so3",
    )
    dropped_out = ielin_dropped(test_sp_ten)
    pseudo_loss = dropped_out.ten.pow(2).sum()
    grad_ten = torch.autograd.grad(
        outputs=pseudo_loss, inputs=test_sp_ten.ten, retain_graph=True
    )
    assert torch.any(grad_ten[0].abs().bool())
    assert not torch.all(grad_ten[0].abs().bool())
    pseudo_loss.backward()
    assert torch.all(ielin_dropped.linears.grad.abs().sum((1, 2)).bool())

    # Test Norm contraction backwards
    test_sp_ten.ten.grad.data.zero_()
    ielin_dropped.linears.grad.data.zero_()
    dropped_out = ielin_dropped(test_sp_ten)
    pseudo_loss = dropped_out.invariant().sum()
    grad_ten = torch.autograd.grad(
        outputs=pseudo_loss, inputs=test_sp_ten.ten, retain_graph=True
    )
    assert torch.any(grad_ten[0].abs().bool())
    assert not torch.all(grad_ten[0].abs().bool())
    pseudo_loss.backward()
    assert torch.all(ielin_dropped.linears.grad.abs().sum((1, 2)).bool())


def test_IELin_forward_2d():
    metadata = torch.LongTensor([[8, 4, 2, 4, 1], [8, 4, 2, 4, 1]])
    test_sp_ten = SphericalTensor(
        torch.rand(4, 6, 12, 67, 67),
        (
            3,
            4,
        ),
        metadata,
    )
    ielin_dense = IELin(
        torch.LongTensor([8, 4, 2, 4, 1]),
        torch.LongTensor([2, 3, 4, 5, 6]),
        group="so3",
    )
    out1 = ielin_dense(test_sp_ten)
    assert out1.ten.shape == (4, 6, 12, 67, 120)
    assert out1.num_channels == (
        19,
        20,
    )
    assert torch.all(
        out1.metadata.eq(torch.LongTensor([[8, 4, 2, 4, 1], [2, 3, 4, 5, 6]]))
    )

    out2 = ielin_dense(out1.transpose_repdims(inplace=True))
    assert out2.ten.shape == (4, 6, 12, 120, 120)
    assert out2.num_channels == (
        20,
        20,
    )
    assert torch.all(
        out2.metadata.eq(torch.LongTensor([[2, 3, 4, 5, 6], [2, 3, 4, 5, 6]]))
    )


def test_IELin_backward_2d():
    metadata = torch.LongTensor([[8, 4, 2, 4, 1], [8, 4, 2, 4, 1]])
    test_sp_ten = SphericalTensor(
        torch.rand(4, 6, 12, 67, 67),
        (
            3,
            4,
        ),
        metadata,
    )
    test_sp_ten.ten.requires_grad = True
    ielin_dense = IELin(
        torch.LongTensor([8, 4, 2, 4, 1]),
        torch.LongTensor([2, 3, 4, 5, 6]),
        group="so3",
    )
    out1 = ielin_dense(test_sp_ten)
    out2 = ielin_dense(out1.transpose_repdims(inplace=True))

    pseudo_loss = out2.ten.pow(2).sum()
    grad_ten = torch.autograd.grad(
        outputs=pseudo_loss, inputs=test_sp_ten.ten, retain_graph=True
    )
    assert grad_ten[0].shape == (4, 6, 12, 67, 67)
    assert torch.any(grad_ten[0].abs().bool())
    assert torch.all(grad_ten[0].abs().bool())
    pseudo_loss.backward()
    assert torch.all(ielin_dense.linears.grad.abs().sum((1, 2)).bool())

    # Test Norm contraction backwards
    test_sp_ten.ten.grad.data.zero_()
    ielin_dense.linears.grad.data.zero_()
    out1 = ielin_dense(test_sp_ten)
    out2 = ielin_dense(out1.transpose_repdims(inplace=True))
    pseudo_loss = out2.invariant().sum() - 0.3
    grad_ten = torch.autograd.grad(
        outputs=pseudo_loss, inputs=test_sp_ten.ten, retain_graph=True
    )
    assert grad_ten[0].shape == (4, 6, 12, 67, 67)
    assert torch.any(grad_ten[0].abs().bool())
    assert torch.all(grad_ten[0].abs().bool())
    pseudo_loss.backward()
    assert torch.all(ielin_dense.linears.grad.abs().sum((1, 2)).bool())


def test_RepNorm1d_forward():
    metadata = torch.LongTensor([[8, 4, 2, 4, 1]])
    test_sp_batched = SphericalTensor(torch.rand(1024, 67), (1,), metadata)
    activation = torch.nn.Tanh()
    norm_linear = torch.nn.Linear(19, 20)
    gauge_linear = IELin(
        torch.LongTensor([8, 4, 2, 4, 1]),
        torch.LongTensor([2, 3, 4, 5, 6]),
        group="so3",
    )
    repnorm = RepNorm1d(19)

    n, g = repnorm(test_sp_batched)
    out = gauge_linear(g).scalar_mul(activation(norm_linear(n)), inplace=True)
    assert out.ten.shape == (1024, 120)
    assert torch.all(out.metadata.eq(torch.LongTensor([[2, 3, 4, 5, 6]])))


def test_RepNorm1d_backward():
    metadata = torch.LongTensor([[8, 4, 2, 4, 1]])
    test_sp_batched = SphericalTensor(torch.rand(1024, 67), (1,), metadata)
    test_sp_batched.ten.requires_grad = True
    activation = torch.nn.Tanh()
    norm_linear = torch.nn.Linear(19, 20)
    gauge_linear = IELin(
        torch.LongTensor([8, 4, 2, 4, 1]),
        torch.LongTensor([2, 3, 4, 5, 6]),
        group="so3",
    )
    repnorm = RepNorm1d(19)

    n, g = repnorm(test_sp_batched)
    out = gauge_linear(g).scalar_mul(activation(norm_linear(n)), inplace=True)

    pseudo_loss = out.invariant().pow(2).sum() - 0.3
    pseudo_loss.backward()
    assert torch.all(test_sp_batched.ten.grad.abs().bool())
    assert torch.all(norm_linear.weight.grad.abs().bool())
    assert torch.all(norm_linear.bias.grad.abs().bool())
    assert torch.all(gauge_linear.linears.grad.abs().sum((1, 2)).bool())


def test_kernel_broadcast():
    rshmodule = RSHxyz(max_l=6)
    xyz = torch.normal(0, 1, size=(64, 3))
    rshs = rshmodule(xyz)
    feat = torch.rand(64, 128)
    kbd = KernelBroadcast(torch.LongTensor([64, 32, 16, 8, 4, 2, 2]))
    out = kbd(rshs, feat)
    assert torch.all(out != 0)
