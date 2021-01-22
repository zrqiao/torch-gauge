"""
Testing real spherical harmonics generation
"""

import math
import random

import torch

from torch_gauge.o3.rsh import RSHxyz
from torch_gauge.o3.wigner import wigner_D_rsh


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


def test_wigner_rsh_z():
    for _ in range(10):
        alpha = random.random() * 2 * math.pi
        rsh_wigner_1 = wigner_D_rsh(1, alpha, 0, 0)
        ref_rotmat = torch.tensor(
            [
                [math.cos(alpha), 0, -math.sin(alpha)],
                [0, 1, 0],
                [math.sin(alpha), 0, math.cos(alpha)],
            ],
            dtype=torch.double,
        ).t()
        assert torch.allclose(rsh_wigner_1, ref_rotmat, atol=1e-7, rtol=1e-5,), print(
            "Wigner-D at j=1 does not match the Cartesian rotation matrix: ",
            rsh_wigner_1,
            ref_rotmat,
        )
