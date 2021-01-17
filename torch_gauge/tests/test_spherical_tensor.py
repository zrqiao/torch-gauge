import torch

from torch_gauge.o3.spherical import SphericalTensor


def test_spherical_tensor_creation1d():
    dten = torch.rand(4, 6, 12, 101, 7)
    metadata = torch.LongTensor([[7, 9, 1, 5, 3]])
    SphericalTensor(dten, (3,), metadata)
    return 0


def test_spherical_tensor_creation2d():
    dten = torch.rand(4, 6, 12, 101, 7)
    metadata = torch.LongTensor([[1, 2, 1, 0, 0], [7, 9, 1, 5, 3]])
    SphericalTensor(dten, (2, 3), metadata)
    return 0


def test_spherical_tensor_layout1d():
    dten = torch.rand(4, 19)
    metadata = torch.LongTensor([[1, 2, 1, 1]])
    test_sp_ten = SphericalTensor(dten, (1,), metadata)
    assert test_sp_ten.num_channels == (5,)
    assert torch.all(
        test_sp_ten.rep_layout[0][0].eq(
            torch.LongTensor([0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3])
        )
    )
    assert torch.all(
        test_sp_ten.rep_layout[0][1].eq(
            torch.LongTensor([0, 0, 0, 1, 1, 2, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6])
        )
    )
    assert torch.all(
        test_sp_ten.rep_layout[0][2].eq(
            torch.LongTensor([0, 1, 2, 1, 2, 1, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4])
        )
    )
    return 0
