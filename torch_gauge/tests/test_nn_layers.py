"""
Testing Spherical neural operations
"""

import torch

from torch_gauge.nn import IELin
from torch_gauge.o3.spherical import SphericalTensor


def test_IELin_forward_1d():
    metadata = torch.LongTensor([[8, 4, 2, 4, 1]])
    test_sp_ten = SphericalTensor(torch.rand(4, 6, 12, 67), (3,), metadata)
    ielin_dense = IELin(
        torch.LongTensor([8, 4, 2, 4, 1]), torch.LongTensor([2, 3, 4, 5, 6])
    )
    dense_out = ielin_dense(test_sp_ten)
    assert dense_out.ten.shape == (4, 6, 12, 120)
    assert dense_out.num_channels == (120,)
    assert torch.all(dense_out.metadata.eq(torch.LongTensor([[2, 3, 4, 5, 6]])))

    ielin_dropped = IELin(
        torch.LongTensor([8, 4, 2, 4, 1]), torch.LongTensor([2, 0, 1, 3, 0])
    )
    dropped_out = ielin_dropped(test_sp_ten)
    assert dropped_out.ten.shape == (4, 6, 12, 28)
    assert dropped_out.num_channels == (28,)
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
    grad_ten = torch.autograd.grad(outputs=pseudo_loss, inputs=test_sp_ten.ten, retain_graph=True)
    assert torch.any(grad_ten[0].abs().bool())
    assert not torch.all(grad_ten[0].abs().bool())
    pseudo_loss.backward()
    for linear in ielin_dropped.linears:
        if linear is not None:
            assert torch.all(linear.weight.grad.abs().bool())
