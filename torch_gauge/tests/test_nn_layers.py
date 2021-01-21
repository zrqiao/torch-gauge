"""
Testing Spherical neural operations
"""

import torch

from torch_gauge.nn import IELin, RepNorm1d
from torch_gauge.o3.spherical import SphericalTensor


def test_IELin_forward_1d():
    metadata = torch.LongTensor([[8, 4, 2, 4, 1]])
    test_sp_ten = SphericalTensor(torch.rand(4, 6, 12, 67), (3,), metadata)
    ielin_dense = IELin(
        torch.LongTensor([8, 4, 2, 4, 1]), torch.LongTensor([2, 3, 4, 5, 6])
    )
    dense_out = ielin_dense(test_sp_ten)
    assert dense_out.ten.shape == (4, 6, 12, 120)
    assert dense_out.num_channels == (20,)
    assert torch.all(dense_out.metadata.eq(torch.LongTensor([[2, 3, 4, 5, 6]])))

    ielin_dropped = IELin(
        torch.LongTensor([8, 4, 2, 4, 1]), torch.LongTensor([2, 0, 1, 3, 0])
    )
    dropped_out = ielin_dropped(test_sp_ten)
    assert dropped_out.ten.shape == (4, 6, 12, 28)
    assert dropped_out.num_channels == (6,)
    assert torch.all(dropped_out.metadata.eq(torch.LongTensor([[2, 0, 1, 3, 0]])))


def test_IELin_backward_1d():
    metadata = torch.LongTensor([[8, 4, 2, 4, 1]])
    test_sp_ten = SphericalTensor(torch.rand(4, 6, 12, 67), (3,), metadata)
    test_sp_ten.ten.requires_grad = True
    ielin_dense = IELin(
        torch.LongTensor([8, 4, 2, 4, 1]), torch.LongTensor([2, 3, 4, 5, 6])
    )
    dense_out = ielin_dense(test_sp_ten)
    pseudo_loss = dense_out.ten.pow(2).sum()
    grad_ten = torch.autograd.grad(outputs=pseudo_loss, inputs=test_sp_ten.ten)
    assert torch.all(grad_ten[0].abs().bool())

    ielin_dropped = IELin(
        torch.LongTensor([8, 4, 2, 4, 1]), torch.LongTensor([2, 0, 1, 3, 0])
    )
    dropped_out = ielin_dropped(test_sp_ten)
    pseudo_loss = dropped_out.ten.pow(2).sum()
    grad_ten = torch.autograd.grad(
        outputs=pseudo_loss, inputs=test_sp_ten.ten, retain_graph=True
    )
    assert torch.any(grad_ten[0].abs().bool())
    assert not torch.all(grad_ten[0].abs().bool())
    pseudo_loss.backward()
    for linear in ielin_dropped.linears:
        if linear is not None:
            assert torch.all(linear.weight.grad.abs().bool())

    # Test Norm contraction backwards
    test_sp_ten.ten.grad.data.zero_()
    for linear in ielin_dropped.linears:
        if linear is not None:
            linear.weight.grad.data.zero_()
    dropped_out = ielin_dropped(test_sp_ten)
    pseudo_loss = dropped_out.invariant().sum()
    grad_ten = torch.autograd.grad(
        outputs=pseudo_loss, inputs=test_sp_ten.ten, retain_graph=True
    )
    assert torch.any(grad_ten[0].abs().bool())
    assert not torch.all(grad_ten[0].abs().bool())
    pseudo_loss.backward()
    for linear in ielin_dropped.linears:
        if linear is not None:
            assert torch.all(linear.weight.grad.abs().bool())


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
        torch.LongTensor([8, 4, 2, 4, 1]), torch.LongTensor([2, 3, 4, 5, 6])
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
        torch.LongTensor([8, 4, 2, 4, 1]), torch.LongTensor([2, 3, 4, 5, 6])
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
    for linear in ielin_dense.linears:
        assert torch.all(linear.weight.grad.abs().bool())

    # Test Norm contraction backwards
    test_sp_ten.ten.grad.data.zero_()
    for linear in ielin_dense.linears:
        linear.weight.grad.data.zero_()
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
    for linear in ielin_dense.linears:
        assert torch.all(linear.weight.grad.abs().bool())


def test_RepNorm1d_forward():
    metadata = torch.LongTensor([[8, 4, 2, 4, 1]])
    test_sp_batched = SphericalTensor(torch.rand(1024, 67), (1,), metadata)
    activation = torch.nn.Tanh()
    norm_linear = torch.nn.Linear(19, 20)
    gauge_linear = IELin(
        torch.LongTensor([8, 4, 2, 4, 1]), torch.LongTensor([2, 3, 4, 5, 6])
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
        torch.LongTensor([8, 4, 2, 4, 1]), torch.LongTensor([2, 3, 4, 5, 6])
    )
    repnorm = RepNorm1d(19)

    n, g = repnorm(test_sp_batched)
    out = gauge_linear(g).scalar_mul(activation(norm_linear(n)), inplace=True)

    pseudo_loss = out.invariant().pow(2).sum() - 0.3
    pseudo_loss.backward()
    assert torch.all(test_sp_batched.ten.grad.abs().bool())
    assert torch.all(norm_linear.weight.grad.abs().bool())
    assert torch.all(norm_linear.bias.grad.abs().bool())
    for linear in gauge_linear.linears:
        assert torch.all(linear.weight.grad.abs().bool())
