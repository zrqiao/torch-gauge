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


def test_spherical_tensor_scalar_product():
    dten = torch.tensor(
        [
            [1.1, 2.2, 0.5, 0.6, -0.6, 0.9, 0.7, 0.3, 0.1, 0.2, 0.3, 0.4, 0.5],
            [0., 0., 0.2, 0., 0.2, 0., 0.3, 0., 1., 1., 1., 1., 1.],
        ]
    )
    metadata = torch.LongTensor([[2, 2, 1]])
    test_sp_ten = SphericalTensor(dten, (1,), metadata)
    scale_ten = torch.tensor([[2., 0.5, 3., 4., 1.], [101., 7., 8., 82., 4.]])
    dten_outplace = test_sp_ten.scalar_mul(scale_ten)
    assert torch.allclose(
        dten_outplace.ten,
        torch.tensor(
            [
                [2.2, 1.1, 1.5, 2.4, -1.8, 3.6, 2.1, 1.2, 0.1, 0.2, 0.3, 0.4, 0.5],
                [0., 0., 1.6, 0., 1.6, 0., 2.4, 0., 4., 4., 4., 4., 4.],
            ]
        ),
        atol=1e-10,
        rtol=1e-7,
    )
    assert torch.all(dten_outplace.rep_layout[0].eq(test_sp_ten.rep_layout[0]))
    dten_inplace = test_sp_ten.scalar_mul(scale_ten, inplace=True)
    assert torch.allclose(
        dten_inplace.ten,
        torch.tensor(
            [
                [2.2, 1.1, 1.5, 2.4, -1.8, 3.6, 2.1, 1.2, 0.1, 0.2, 0.3, 0.4, 0.5],
                [0., 0., 1.6, 0., 1.6, 0., 2.4, 0., 4., 4., 4., 4., 4.],
            ]
        ),
        atol=1e-10,
        rtol=1e-7,
    )
    assert torch.all(dten_inplace.rep_layout[0].eq(test_sp_ten.rep_layout[0]))


def test_spherical_tensor_dot_product():
    dten1 = torch.rand(4, 6, 1, 101, 7)
    metadata = torch.LongTensor([[7, 9, 1, 5, 3]])
    test1 = SphericalTensor(dten1, (3,), metadata)
    dten2 = torch.rand(4, 6, 12, 101, 7)
    metadata = torch.LongTensor([[1, 2, 1, 0, 0], [7, 9, 1, 5, 3]])
    test2 = SphericalTensor(dten2, (2, 3), metadata)
    test_dot = test2.dot(test1, dim=3)
    assert torch.all(
        test_dot.ten.eq(test1.ten.mul(test2.ten).sum(3))
    )
    assert test_dot.ten.shape == (4, 6, 12, 7)
    assert test_dot.rep_dims == (2,)
    assert torch.all(test_dot.metadata.eq(torch.LongTensor([[1, 2, 1, 0, 0]])))
    assert torch.all(test_dot.rep_layout[0].eq(test2.rep_layout[0]))


def test_spherical_tensor_rep_dot():
    # Minimal 1d example
    metadata = torch.LongTensor([[1, 2]])
    test_1d_1 = SphericalTensor(torch.tensor([0.2, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4]), (0,), metadata)
    test_1d_2 = SphericalTensor(torch.tensor([0.4, 0.2, 0.3, -0.4, -0.4, -0.3, 0.4]), (0,), metadata)
    test_1d_out = test_1d_1.rep_dot(test_1d_2, dim=0)
    assert torch.allclose(
        test_1d_out,
        torch.tensor([0.08, -0.19, 0.14]),
        atol=1e-9,
        rtol=1e-7,
    )

    # 2d shape test
    dten1 = torch.rand(4, 6, 1, 101, 7)
    metadata = torch.LongTensor([[7, 9, 1, 5, 3]])
    test1 = SphericalTensor(dten1, (3,), metadata)
    dten2 = torch.rand(4, 6, 12, 101, 7)
    metadata = torch.LongTensor([[1, 2, 1, 0, 0], [7, 9, 1, 5, 3]])
    test2 = SphericalTensor(dten2, (2, 3), metadata)
    test_dot = test2.rep_dot(test1, dim=3)
    assert test_dot.ten.shape == (4, 6, 12, 25, 7)
    assert test_dot.rep_dims == (2,)
    assert torch.all(test_dot.metadata.eq(torch.LongTensor([[1, 2, 1, 0, 0]])))
    assert torch.all(test_dot.rep_layout[0].eq(test2.rep_layout[0]))

    # When L=0 entries are positive, self rep-dot should return a
    # tensor almost the same as the invariant content within threshold _norm_eps
    test3 = SphericalTensor(torch.rand(32, 11, 5, 101, 7), (3,), torch.LongTensor([[7, 9, 1, 5, 3]]))
    dot_3 = test3.rep_dot(test3, dim=3)
    assert torch.allclose(
        dot_3,
        test3.invariant().pow(2),
        atol=1e-4,
        rtol=1e-3,
    )
