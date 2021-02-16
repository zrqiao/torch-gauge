import math
import random

import torch

from torch_gauge.o3 import O3Tensor
from torch_gauge.o3.clebsch_gordan import CGPCoupler
from torch_gauge.o3.wigner import wigner_D_rsh

torch.manual_seed(42)


def test_cgp_selection_rule():
    # max_l = 1, same layout
    metadata = torch.LongTensor([[24, 0, 12, 0]])
    dten1 = torch.rand(4, 6, 2, 60, 7)
    dten2 = torch.rand(4, 6, 2, 60, 7)
    spten1 = O3Tensor(dten1, (3,), metadata)
    spten2 = O3Tensor(dten2, (3,), metadata)
    coupler2 = CGPCoupler(metadata[0], metadata[0], trunc_in=False, dtype=torch.float)
    cat_out = coupler2(spten1, spten2)
    assert cat_out.ten.shape == (4, 6, 2, 108, 7)
    assert torch.all(cat_out.metadata.eq(torch.LongTensor([[36, 0, 24, 0]])))
    coupler3 = CGPCoupler(metadata[0], metadata[0], trunc_in=True, dtype=torch.float)
    trunc_out = coupler3(spten1, spten2)
    assert trunc_out.ten.shape == (4, 6, 2, 96, 7)
    assert torch.all(trunc_out.metadata.eq(torch.LongTensor([[24, 0, 24, 0]])))

    # max_l = 1, hetero layout
    metadata1 = torch.LongTensor([[24, 0, 12, 0]])
    metadata2 = torch.LongTensor([[17, 3, 5, 15]])
    dten1 = torch.rand(4, 6, 2, 60, 7)
    dten2 = torch.rand(4, 6, 2, 80, 7)
    spten1 = O3Tensor(dten1, (3,), metadata1)
    spten2 = O3Tensor(dten2, (3,), metadata2)
    coupler2 = CGPCoupler(metadata1[0], metadata2[0], trunc_in=False, dtype=torch.float)
    cat_out = coupler2(spten1, spten2)
    assert torch.all(cat_out.metadata.eq(torch.LongTensor([[22, 6, 29, 23]])))
    assert cat_out.ten.shape == (4, 6, 2, 184, 7)
    coupler3 = CGPCoupler(metadata1[0], metadata2[0], trunc_in=True, dtype=torch.float)
    trunc_out = coupler3(spten1, spten2)
    assert torch.all(trunc_out.metadata.eq(torch.LongTensor([[17, 3, 17, 18]])))
    assert trunc_out.ten.shape == (4, 6, 2, 125, 7)

    # Large l coupling
    metadata1 = torch.LongTensor(
        [[128, 64, 64, 32, 32, 16, 16, 8, 8, 4, 4, 2, 2, 1, 1, 0, 1, 0]]
    )
    metadata2 = torch.LongTensor(
        [[160, 80, 80, 40, 40, 20, 20, 10, 10, 5, 5, 1, 1, 0, 0, 0, 0, 0]]
    )
    dten1 = torch.rand(8, 1133)
    dten2 = torch.rand(8, 1324)
    spten1 = O3Tensor(dten1, (1,), metadata1)
    spten2 = O3Tensor(dten2, (1,), metadata2)
    coupler2 = CGPCoupler(metadata1[0], metadata2[0], trunc_in=False, dtype=torch.float)
    coupler3 = CGPCoupler(metadata1[0], metadata2[0], trunc_in=True, dtype=torch.float)

    coupling_out = coupler2(spten1, spten2)
    assert torch.all(
        coupling_out.metadata.eq(
            torch.LongTensor(
                [
                    [
                        378,
                        283,
                        561,
                        503,
                        558,
                        455,
                        438,
                        339,
                        306,
                        219,
                        188,
                        106,
                        89,
                        59,
                        48,
                        0,
                        37,
                        0,
                    ]
                ]
            )
        )
    )
    assert not torch.any(coupling_out.ten == 0)
    coupling_out = coupler3(spten1, spten2)
    assert torch.all(
        coupling_out.metadata.eq(
            torch.LongTensor(
                [
                    [
                        290,
                        196,
                        414,
                        252,
                        356,
                        200,
                        253,
                        130,
                        160,
                        75,
                        90,
                        33,
                        44,
                        13,
                        22,
                        0,
                        11,
                        0,
                    ]
                ]
            )
        )
    )
    assert not torch.any(coupling_out.ten == 0)


metadata1 = torch.LongTensor(
    [[128, 64, 64, 32, 32, 16, 16, 8, 8, 4, 4, 2, 2, 1, 1, 0, 1, 0]]
)
metadata2 = torch.LongTensor(
    [[160, 80, 80, 40, 40, 20, 20, 10, 10, 5, 5, 1, 1, 0, 0, 0, 0, 0]]
)
dten1 = torch.rand(48, 1133)
dten2 = torch.rand(48, 1324)
spten1 = O3Tensor(dten1, (1,), metadata1)
spten2 = O3Tensor(dten2, (1,), metadata2)
coupler1 = CGPCoupler(metadata1[0], metadata2[0], trunc_in=False, dtype=torch.float)
coupler2 = CGPCoupler(metadata1[0], metadata2[0], trunc_in=True, dtype=torch.float)


def test_benchmark_cgpcoupler_setup1(benchmark):
    benchmark(coupler1, spten1, spten2)


def test_benchmark_cgpcoupler_setup2(benchmark):
    benchmark(coupler2, spten1, spten2)


def real_wigner_rot(o3ten, alpha, beta, gamma):
    # Naive implementation, do not use in models
    rot_ten = torch.zeros_like(o3ten.ten)
    for l in range(o3ten.metadata.shape[1] // 2):
        for p in (1, -1):
            print(f"Rotating rep {l}, parity {p}...")
            real_wigner_l = wigner_D_rsh(l, alpha, beta, gamma).type(o3ten.ten.dtype)
            l_mask = o3ten.rep_layout[0][0] == l
            p_mask = o3ten.rep_layout[0][3] == p
            torot = o3ten.ten[:, l_mask * p_mask]
            if torot.numel() == 0:
                continue
            torot = torot.view(o3ten.ten.shape[0], 2 * l + 1, -1)
            rot_ten[:, l_mask * p_mask] = (
                torch.einsum("ijk,ja->iak", torot, real_wigner_l)
                .contiguous()
                .view(torot.shape[0], -1)
            )
    return o3ten.self_like(rot_ten)


def condon_shortley_inv(o3ten):
    # Total parity = (-1)**l * p
    cs_phase = torch.pow(-1, o3ten.rep_layout[0][0]) * o3ten.rep_layout[0][3]
    return o3ten.self_like(o3ten.ten.mul(cs_phase))


# Equivariance tests w.r.t improper rotations
alpha = random.random() * 2 * math.pi
beta = random.random() * 2 * math.pi
gamma = random.random() * 2 * math.pi


def test_cgp_coupling_equivariance1():
    out1: O3Tensor = coupler1(spten1, spten2)
    out1_rot = condon_shortley_inv(real_wigner_rot(out1, alpha, beta, gamma))

    spten1_rot = condon_shortley_inv(real_wigner_rot(spten1, alpha, beta, gamma))
    spten2_rot = condon_shortley_inv(real_wigner_rot(spten2, alpha, beta, gamma))
    rot_out1 = coupler1(spten1_rot, spten2_rot)
    assert not torch.allclose(out1_rot.ten, out1.ten, atol=1e-3), print(
        (out1_rot.ten - out1.ten).flatten().abs()
    )
    print(out1.ten)
    print(out1_rot.ten)
    print(rot_out1.ten)
    assert torch.allclose(rot_out1.ten, out1_rot.ten, atol=1e-5, rtol=1e-5), print(
        (out1_rot.ten - rot_out1.ten).flatten().abs(), out1_rot.metadata
    )


def test_cgp_coupling_equivariance2():
    out2: O3Tensor = coupler2(spten1, spten2)
    out2_rot = condon_shortley_inv(real_wigner_rot(out2, alpha, beta, gamma))

    spten1_rot = condon_shortley_inv(real_wigner_rot(spten1, alpha, beta, gamma))
    spten2_rot = condon_shortley_inv(real_wigner_rot(spten2, alpha, beta, gamma))
    rot_out2 = coupler2(spten1_rot, spten2_rot)
    assert not torch.allclose(out2_rot.ten, out2.ten, atol=1e-3), print(
        (out2_rot.ten - out2.ten).flatten().abs()
    )
    assert torch.allclose(rot_out2.ten, out2_rot.ten, atol=1e-5, rtol=1e-5), print(
        (out2_rot.ten - rot_out2.ten).flatten().abs()
    )
