import math
import random

import torch

from torch_gauge.o3 import SphericalTensor
from torch_gauge.o3.clebsch_gordan import (
    CGCoupler,
    LeviCivitaCoupler,
    get_clebsch_gordan_coefficient,
    get_rsh_cg_coefficients,
)
from torch_gauge.o3.wigner import wigner_D_rsh

random.seed(42)
torch.manual_seed(42)


def test_generate_cg_csh():
    max_j = 4
    for j1 in range(max_j + 1):
        for j2 in range(max_j + 1):
            for j in range(abs(j1 - j2), max_j + 1):
                print(
                    f"Generating Clebsch-Gordan coefficients for j1={j1}, j2={j2}, j={j}"
                )
                for m1 in range(-j1, j1 + 1):
                    for m2 in range(-j2, j2 + 1):
                        get_clebsch_gordan_coefficient(j1, j2, j, m1, m2, m1 + m2)


def test_levi_civita():
    metadata = torch.LongTensor([[24, 12]])
    dten1 = torch.rand(4, 6, 2, 60, 7)
    dten2 = torch.rand(4, 6, 2, 60, 7)
    spten1 = SphericalTensor(dten1, (3,), metadata)
    spten2 = SphericalTensor(dten2, (3,), metadata)
    coupler = LeviCivitaCoupler(metadata[0])
    overlap_out = coupler(spten1, spten2, overlap_out=True)
    assert overlap_out.ten.shape == (4, 6, 2, 60, 7)
    assert torch.all(overlap_out.metadata.eq(torch.LongTensor([[24, 12]])))
    cat_out = coupler(spten1, spten2, overlap_out=False)
    assert cat_out.ten.shape == (4, 6, 2, 144, 7)
    assert torch.all(cat_out.metadata.eq(torch.LongTensor([[36, 36]])))


def test_rsh_cg_generation():
    for j1 in range(4):
        for j2 in range(4):
            for j in range(4):
                get_rsh_cg_coefficients(j1, j2, j)


def test_cg_selection_rule():
    # max_l = 1, same layout
    metadata = torch.LongTensor([[24, 12]])
    dten1 = torch.rand(4, 6, 2, 60, 7)
    dten2 = torch.rand(4, 6, 2, 60, 7)
    spten1 = SphericalTensor(dten1, (3,), metadata)
    spten2 = SphericalTensor(dten2, (3,), metadata)
    coupler1 = CGCoupler(
        metadata[0], metadata[0], overlap_out=True, trunc_in=False, dtype=torch.float
    )
    overlap_out = coupler1(spten1, spten2)
    assert overlap_out.ten.shape == (4, 6, 2, 60, 7)
    assert torch.all(overlap_out.metadata.eq(torch.LongTensor([[24, 12]])))
    coupler2 = CGCoupler(
        metadata[0], metadata[0], overlap_out=False, trunc_in=False, dtype=torch.float
    )
    cat_out = coupler2(spten1, spten2)
    assert cat_out.ten.shape == (4, 6, 2, 144, 7)
    assert torch.all(cat_out.metadata.eq(torch.LongTensor([[36, 36]])))
    coupler3 = CGCoupler(
        metadata[0], metadata[0], overlap_out=False, trunc_in=True, dtype=torch.float
    )
    trunc_out = coupler3(spten1, spten2)
    assert trunc_out.ten.shape == (4, 6, 2, 96, 7)
    assert torch.all(trunc_out.metadata.eq(torch.LongTensor([[24, 24]])))
    coupler4 = CGCoupler(
        metadata[0], metadata[0], overlap_out=True, trunc_in=True, dtype=torch.float
    )
    trunc_overlap_out = coupler4(spten1, spten2)
    assert trunc_overlap_out.ten.shape == (4, 6, 2, 60, 7)
    assert torch.all(trunc_overlap_out.metadata.eq(torch.LongTensor([[24, 12]])))

    # max_l = 1, hetero layout
    metadata1 = torch.LongTensor([[24, 12]])
    metadata2 = torch.LongTensor([[17, 21]])
    dten1 = torch.rand(4, 6, 2, 60, 7)
    dten2 = torch.rand(4, 6, 2, 80, 7)
    spten1 = SphericalTensor(dten1, (3,), metadata1)
    spten2 = SphericalTensor(dten2, (3,), metadata2)
    coupler1 = CGCoupler(
        metadata1[0], metadata2[0], overlap_out=True, trunc_in=False, dtype=torch.float
    )
    overlap_out = coupler1(spten1, spten2)
    assert overlap_out.ten.shape == (4, 6, 2, 80, 7)
    assert torch.all(overlap_out.metadata.eq(torch.LongTensor([[17, 21]])))
    coupler2 = CGCoupler(
        metadata1[0], metadata2[0], overlap_out=False, trunc_in=False, dtype=torch.float
    )
    cat_out = coupler2(spten1, spten2)
    assert cat_out.ten.shape == (4, 6, 2, 164, 7)
    assert torch.all(cat_out.metadata.eq(torch.LongTensor([[29, 45]])))
    coupler3 = CGCoupler(
        metadata1[0], metadata2[0], overlap_out=False, trunc_in=True, dtype=torch.float
    )
    trunc_out = coupler3(spten1, spten2)
    assert trunc_out.ten.shape == (4, 6, 2, 116, 7)
    assert torch.all(trunc_out.metadata.eq(torch.LongTensor([[17, 33]])))
    coupler4 = CGCoupler(
        metadata1[0], metadata2[0], overlap_out=True, trunc_in=True, dtype=torch.float
    )
    trunc_overlap_out = coupler4(spten1, spten2)
    assert trunc_overlap_out.ten.shape == (4, 6, 2, 80, 7)
    assert torch.all(trunc_overlap_out.metadata.eq(torch.LongTensor([[17, 21]])))

    # Large l coupling
    metadata1 = torch.LongTensor([[128, 64, 32, 16, 8, 4, 2, 1, 1]])
    metadata2 = torch.LongTensor([[160, 80, 40, 20, 10, 5, 1, 0, 0]])
    dten1 = torch.rand(8, 766)
    dten2 = torch.rand(8, 898)
    spten1 = SphericalTensor(dten1, (1,), metadata1)
    spten2 = SphericalTensor(dten2, (1,), metadata2)
    coupler1 = CGCoupler(
        metadata1[0], metadata2[0], overlap_out=True, trunc_in=False, dtype=torch.float
    )
    coupler2 = CGCoupler(
        metadata1[0], metadata2[0], overlap_out=False, trunc_in=False, dtype=torch.float
    )
    coupler3 = CGCoupler(
        metadata1[0], metadata2[0], overlap_out=False, trunc_in=True, dtype=torch.float
    )
    coupler4 = CGCoupler(
        metadata1[0], metadata2[0], overlap_out=True, trunc_in=True, dtype=torch.float
    )

    coupling_out = coupler1(spten1, spten2)
    assert torch.all(
        coupling_out.metadata.eq(torch.LongTensor([[128, 80, 40, 20, 10, 5, 2, 1, 1]]))
    )
    assert not torch.any(coupling_out.ten == 0)
    coupling_out = coupler2(spten1, spten2)
    assert torch.all(
        coupling_out.metadata.eq(
            torch.LongTensor([[253, 408, 380, 292, 200, 124, 60, 34, 28]])
        )
    )
    assert not torch.any(coupling_out.ten == 0)
    coupling_out = coupler3(spten1, spten2)
    assert torch.all(
        coupling_out.metadata.eq(
            torch.LongTensor([[181, 249, 203, 139, 87, 50, 25, 13, 7]])
        )
    )
    assert not torch.any(coupling_out.ten == 0)
    coupling_out = coupler4(spten1, spten2)
    assert torch.all(
        coupling_out.metadata.eq(torch.LongTensor([[128, 80, 40, 20, 10, 5, 2, 1, 1]]))
    )
    assert not torch.any(coupling_out.ten == 0)


metadata1 = torch.LongTensor([[128, 64, 32, 16, 8, 4, 2, 1, 1]])
metadata2 = torch.LongTensor([[160, 80, 40, 20, 10, 5, 1, 0, 0]])
dten1 = torch.rand(64, 766)
dten2 = torch.rand(64, 898)
spten1 = SphericalTensor(dten1, (1,), metadata1)
spten2 = SphericalTensor(dten2, (1,), metadata2)
coupler1 = CGCoupler(
    metadata1[0], metadata2[0], overlap_out=True, trunc_in=False, dtype=torch.float
)
coupler2 = CGCoupler(
    metadata1[0], metadata2[0], overlap_out=False, trunc_in=False, dtype=torch.float
)
coupler3 = CGCoupler(
    metadata1[0], metadata2[0], overlap_out=False, trunc_in=True, dtype=torch.float
)
coupler4 = CGCoupler(
    metadata1[0], metadata2[0], overlap_out=True, trunc_in=True, dtype=torch.float
)


def test_benchmark_coupler_setup1(benchmark):
    benchmark(coupler1, spten1, spten2)


def test_benchmark_coupler_setup2(benchmark):
    benchmark(coupler2, spten1, spten2)


def test_benchmark_coupler_setup3(benchmark):
    benchmark(coupler3, spten1, spten2)


def test_benchmark_coupler_setup4(benchmark):
    benchmark(coupler4, spten1, spten2)


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


# Equivariance tests
# Couple-then-rotate must exactly agree with rotate-then-couple
alpha = random.random() * 2 * math.pi
beta = random.random() * 2 * math.pi
gamma = random.random() * 2 * math.pi


def test_cg_coupling_equivariance1():
    out1: SphericalTensor = coupler1(spten1, spten2)
    out1_rot = real_wigner_rot(out1, alpha, beta, gamma)

    spten1_rot = real_wigner_rot(spten1, alpha, beta, gamma)
    spten2_rot = real_wigner_rot(spten2, alpha, beta, gamma)
    rot_out1 = coupler1(spten1_rot, spten2_rot)
    assert not torch.allclose(out1_rot.ten, out1.ten, atol=1e-4, rtol=1e-4), print(
        (out1_rot.ten - out1.ten).flatten().abs()
    )
    assert torch.allclose(rot_out1.ten, out1_rot.ten, atol=1e-5, rtol=1e-5), print(
        (out1_rot.ten - rot_out1.ten).flatten().abs()
    )


def test_cg_coupling_equivariance2():
    out2: SphericalTensor = coupler2(spten1, spten2)
    out2_rot = real_wigner_rot(out2, alpha, beta, gamma)

    spten1_rot = real_wigner_rot(spten1, alpha, beta, gamma)
    spten2_rot = real_wigner_rot(spten2, alpha, beta, gamma)
    rot_out2 = coupler2(spten1_rot, spten2_rot)
    assert not torch.allclose(out2_rot.ten, out2.ten, atol=1e-4, rtol=1e-4), print(
        (out2_rot.ten - out2.ten).flatten().abs()
    )
    assert torch.allclose(rot_out2.ten, out2_rot.ten, atol=1e-5, rtol=1e-5), print(
        (out2_rot.ten - rot_out2.ten).flatten().abs()
    )


def test_cg_coupling_equivariance3():
    out3: SphericalTensor = coupler3(spten1, spten2)
    out3_rot = real_wigner_rot(out3, alpha, beta, gamma)

    spten1_rot = real_wigner_rot(spten1, alpha, beta, gamma)
    spten2_rot = real_wigner_rot(spten2, alpha, beta, gamma)
    rot_out3 = coupler3(spten1_rot, spten2_rot)
    assert not torch.allclose(out3_rot.ten, out3.ten, atol=1e-4, rtol=1e-4), print(
        (out3_rot.ten - out3.ten).flatten().abs()
    )
    assert torch.allclose(rot_out3.ten, out3_rot.ten, atol=1e-5, rtol=1e-5), print(
        (out3_rot.ten - rot_out3.ten).flatten().abs()
    )


def test_cg_coupling_equivariance4():
    out4: SphericalTensor = coupler4(spten1, spten2)
    out4_rot = real_wigner_rot(out4, alpha, beta, gamma)

    spten1_rot = real_wigner_rot(spten1, alpha, beta, gamma)
    spten2_rot = real_wigner_rot(spten2, alpha, beta, gamma)
    rot_out4 = coupler4(spten1_rot, spten2_rot)
    assert not torch.allclose(out4_rot.ten, out4.ten, atol=1e-4, rtol=1e-4), print(
        (out4_rot.ten - out4.ten).flatten().abs()
    )
    assert torch.allclose(rot_out4.ten, out4_rot.ten, atol=1e-5, rtol=1e-5), print(
        (out4_rot.ten - rot_out4.ten).flatten().abs()
    )


def test_cg_coupling_parity_equivariance_polar():
    # Polar tensor, improper rotation
    coupler = CGCoupler(
        metadata1[0],
        metadata2[0],
        parity=1,
        overlap_out=False,
        trunc_in=False,
        dtype=torch.float,
    )
    out: SphericalTensor = coupler(spten1, spten2)
    assert torch.all(
        out.metadata.eq(torch.LongTensor([[253, 283, 241, 175, 117, 71, 34, 19, 16]]))
    )
    out_rot = real_wigner_rot(out, alpha, beta, gamma)
    # Condon-Shortley phase upon inversion
    out_rot.ten *= torch.pow(-1, out_rot.rep_layout[0][0])

    spten1_rot = real_wigner_rot(spten1, alpha, beta, gamma)
    spten1_rot.ten *= torch.pow(-1, spten1_rot.rep_layout[0][0])
    spten2_rot = real_wigner_rot(spten2, alpha, beta, gamma)
    spten2_rot.ten *= torch.pow(-1, spten2_rot.rep_layout[0][0])
    rot_out = coupler(spten1_rot, spten2_rot)
    assert torch.allclose(rot_out.ten, out_rot.ten, atol=1e-5, rtol=1e-5), print(
        (out_rot.ten - rot_out.ten).flatten().abs()
    )


def test_cg_coupling_parity_equivariance_pseudo():
    # Pseudo tensor, improper rotation
    coupler = CGCoupler(
        metadata1[0],
        metadata2[0],
        parity=-1,
        overlap_out=False,
        trunc_in=False,
        dtype=torch.float,
    )
    out: SphericalTensor = coupler(spten1, spten2)
    out_rot = real_wigner_rot(out, alpha, beta, gamma)
    assert torch.all(
        out.metadata.eq(torch.LongTensor([[0, 125, 139, 117, 83, 53, 26, 15, 12]]))
    )
    out_rot.ten *= torch.pow(-1, out_rot.rep_layout[0][0])

    spten1_rot = real_wigner_rot(spten1, alpha, beta, gamma)
    spten1_rot.ten *= torch.pow(-1, spten1_rot.rep_layout[0][0])
    spten2_rot = real_wigner_rot(spten2, alpha, beta, gamma)
    spten2_rot.ten *= torch.pow(-1, spten2_rot.rep_layout[0][0])
    rot_out = coupler(spten1_rot, spten2_rot)
    assert torch.allclose(rot_out.ten.neg(), out_rot.ten, atol=1e-5, rtol=1e-5), print(
        (out_rot.ten - rot_out.ten).flatten().abs()
    )
