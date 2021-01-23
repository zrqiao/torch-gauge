"""
Testing real spherical harmonics generation
"""

import math
import random

import torch

from torch_gauge.geometric import rotation_matrix_xyz
from torch_gauge.o3.rsh import RSHxyz
from torch_gauge.o3.wigner import wigner_D_rsh, wigner_small_d_csh


def test_batch_rsh_generation():
    rshmodule = RSHxyz(max_l=6)
    xyz = torch.normal(0, 1, size=(1024, 3))
    rshs = rshmodule(xyz)
    assert torch.all(rshs.metadata.eq(torch.tensor([[1, 1, 1, 1, 1, 1, 1]])))


def test_rsh_explicit():
    rshmodule = RSHxyz(max_l=2)
    # Compare with the explicit formula up to l=2
    for _ in range(100):
        x = torch.normal(0, 1, size=(1,)).double()
        y = torch.normal(0, 1, size=(1,)).double()
        z = torch.normal(0, 1, size=(1,)).double()
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
        rshs_ref = torch.tensor(
            [
                1,
                y,
                z,
                x,
                math.sqrt(3) * x * y,
                math.sqrt(3) * y * z,
                1 / 2 * (3 * z ** 2 - r ** 2),
                math.sqrt(3) * x * z,
                (math.sqrt(3) / 2) * (x ** 2 - y ** 2),
            ]
        )
        assert torch.allclose(
            rshmodule(torch.stack([x, y, z], dim=1)).ten,
            rshs_ref.unsqueeze(0),
            atol=1e-8,
            rtol=0,
        )


def test_rsh_batch_explicit():
    rshmodule = RSHxyz(max_l=2)
    x = torch.normal(0, 1, size=(1024,)).double()
    y = torch.normal(0, 1, size=(1024,)).double()
    z = torch.normal(0, 1, size=(1024,)).double()
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    rshs_ref = torch.stack(
        [
            torch.ones_like(x),
            y,
            z,
            x,
            math.sqrt(3) * x * y,
            math.sqrt(3) * y * z,
            1 / 2 * (3 * z ** 2 - r ** 2),
            math.sqrt(3) * x * z,
            (math.sqrt(3) / 2) * (x ** 2 - y ** 2),
        ],
        dim=1,
    )
    assert torch.allclose(
        rshmodule(torch.stack([x, y, z], dim=1)).ten,
        rshs_ref,
        atol=1e-8,
        rtol=0,
    )


def test_wigner_small_d():
    # Check against table https://www.wikiwand.com/en/Wigner_D-matrix#/List_of_d-matrix_elements
    theta = random.random() * 2 * math.pi
    ref_mat = torch.tensor(
        [
            [
                1 / 2 * (1 + math.cos(theta)),
                1 / math.sqrt(2) * math.sin(theta),
                1 / 2 * (1 - math.cos(theta)),
            ],
            [
                -1 / math.sqrt(2) * math.sin(theta),
                math.cos(theta),
                1 / math.sqrt(2) * math.sin(theta),
            ],
            [
                1 / 2 * (1 - math.cos(theta)),
                -1 / math.sqrt(2) * math.sin(theta),
                1 / 2 * (1 + math.cos(theta)),
            ],
        ],
        dtype=torch.double,
    )
    assert torch.allclose(
        wigner_small_d_csh(1, theta),
        ref_mat,
        atol=1e-7,
    )


def test_wigner_rsh_z():
    # Along z axis
    for _ in range(10):
        alpha = random.random() * 2 * math.pi
        rsh_wigner_1 = wigner_D_rsh(1, alpha, 0, 0)
        # In the y-z-x order
        ref_rotmat = torch.tensor(
            [
                [math.cos(alpha), 0, -math.sin(alpha)],
                [0, 1, 0],
                [math.sin(alpha), 0, math.cos(alpha)],
            ],
            dtype=torch.double,
        )
        assert torch.allclose(rsh_wigner_1, ref_rotmat, atol=1e-6, rtol=1e-5,), print(
            "Wigner-D at j=1 does not match the Cartesian rotation matrix: ",
            rsh_wigner_1,
            ref_rotmat,
        )


def test_wigner_rsh_y():
    # Along y axis
    for _ in range(10):
        beta = random.random() * 2 * math.pi
        rsh_wigner_1 = wigner_D_rsh(1, 0, beta, 0)
        # In the y-z-x order
        ref_rotmat = torch.tensor(
            [
                [1, 0, 0],
                [0, math.cos(beta), math.sin(beta)],
                [0, -math.sin(beta), math.cos(beta)],
            ],
            dtype=torch.double,
        )
        assert torch.allclose(rsh_wigner_1, ref_rotmat, atol=1e-6, rtol=1e-5,), print(
            "Wigner-D at j=1 does not match the Cartesian rotation matrix: ",
            rsh_wigner_1,
            ref_rotmat,
        )


def test_wigner_rsh_zyz():
    # Test Euler rotation
    # Note that space-fixed rotation (gamma, beta, alpha) == Euler rotation (alpha, beta, gamma),
    # as elaborated in Sec. 3.3, Page 175-177 of J. J. Sakurai
    for _ in range(10):
        alpha = random.random() * 2 * math.pi
        beta = random.random() * 2 * math.pi
        gamma = random.random() * 2 * math.pi
        rsh_wigner_1 = wigner_D_rsh(1, alpha, beta, gamma)
        ref1 = torch.tensor(
            [
                [math.cos(alpha), 0, -math.sin(alpha)],
                [0, 1, 0],
                [math.sin(alpha), 0, math.cos(alpha)],
            ],
            dtype=torch.double,
        )
        ref2 = torch.tensor(
            [
                [1, 0, 0],
                [0, math.cos(beta), math.sin(beta)],
                [0, -math.sin(beta), math.cos(beta)],
            ],
            dtype=torch.double,
        )
        ref3 = torch.tensor(
            [
                [math.cos(gamma), 0, -math.sin(gamma)],
                [0, 1, 0],
                [math.sin(gamma), 0, math.cos(gamma)],
            ],
            dtype=torch.double,
        )
        ref_rotmat = ref3.mm(ref2).mm(ref1)
        assert torch.allclose(rsh_wigner_1, ref_rotmat, atol=1e-6, rtol=1e-5,), print(
            "Wigner-D at j=1 does not match the Cartesian rotation matrix: ",
            rsh_wigner_1,
            ref_rotmat,
        )


def test_wigner_rsh_rotation():
    # Tight checking up to l=10
    rshmodule = RSHxyz(max_l=10)
    xyz = torch.normal(0, 1, size=(1, 3), dtype=torch.double)
    xyz /= xyz.norm(dim=1).unsqueeze(1)

    alpha = random.random() * 2 * math.pi
    beta = random.random() * 2 * math.pi
    gamma = random.random() * 2 * math.pi
    ref1 = rotation_matrix_xyz(alpha, "z", dtype=torch.double).t()
    ref2 = rotation_matrix_xyz(beta, "y", dtype=torch.double).t()
    ref3 = rotation_matrix_xyz(gamma, "z", dtype=torch.double).t()
    ref_rot = ref3.mm(ref2).mm(ref1)
    rot_xyz = xyz.mm(ref_rot)

    rsh_pre = rshmodule(xyz).ten
    rsh_rot = rshmodule(rot_xyz).ten

    wigner_rot_rshs = []
    for l in range(11):
        real_wigner_l = wigner_D_rsh(l, alpha, beta, gamma)
        wigner_rot_rshs.append(rsh_pre[:, l ** 2 : (l + 1) ** 2].mm(real_wigner_l))

    wigner_rot_rshs = torch.cat(wigner_rot_rshs, dim=1)
    assert torch.allclose(rsh_rot, wigner_rot_rshs, atol=1e-6), print(
        rsh_rot, wigner_rot_rshs
    )
