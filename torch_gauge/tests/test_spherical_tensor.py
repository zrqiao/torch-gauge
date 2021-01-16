import torch
from torch_gauge.o3.spherical import SphericalTensor


def test_spherical_tensor_creation1d():
    dten = torch.rand(4, 6, 12, 101, 7)
    metadata = torch.LongTensor([[7, 9, 1, 5, 3]])
    test_sp_ten = SphericalTensor(dten, (3,), metadata)
    return 0


def test_spherical_tensor_creation2d():
    dten = torch.rand(4, 6, 12, 101, 7)
    metadata = torch.LongTensor([[1, 2, 1, 0, 0], [7, 9, 1, 5, 3]])
    test_sp_ten = SphericalTensor(dten, (2, 3), metadata)
    return 0
