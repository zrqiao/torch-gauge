import torch

from torch_gauge.o3.clebsch_gordan import (
    LeviCivitaCoupler,
    CGCoupler,
    get_clebsch_gordan_coefficient,
    get_rsh_cg_coefficients,
)
from torch_gauge.o3.spherical import SphericalTensor


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
    cat_out = coupler(spten1, spten2, overlap_out=False)
    assert cat_out.ten.shape == (4, 6, 2, 144, 7)
    assert torch.all(cat_out.metadata.eq(torch.LongTensor([[36, 36]])))


def test_rsh_cg_generation():
    for j1 in range(4):
        for j2 in range(4):
            for j in range(4):
                get_rsh_cg_coefficients(j1, j2, j)


def test_cg_coupling():
    # max_l = 1
    metadata = torch.LongTensor([[24, 12]])
    dten1 = torch.rand(4, 6, 2, 60, 7)
    dten2 = torch.rand(4, 6, 2, 60, 7)
    spten1 = SphericalTensor(dten1, (3,), metadata)
    spten2 = SphericalTensor(dten2, (3,), metadata)
    coupler = CGCoupler(metadata[0], metadata[0], overlap_out=True, trunc_in=False)
    overlap_out = coupler(spten1, spten2)
    assert overlap_out.ten.shape == (4, 6, 2, 60, 7)
    coupler = CGCoupler(metadata[0], metadata[0], overlap_out=False, trunc_in=False)
    cat_out = coupler(spten1, spten2)
    assert cat_out.ten.shape == (4, 6, 2, 144, 7)
    assert torch.all(cat_out.metadata.eq(torch.LongTensor([[36, 36]])))

