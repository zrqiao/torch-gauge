"""
SO(3) Clebsch-Gordan coefficients and the associated CUDA-compatible coupler modules.

See:
    Sakurai, J. J. "Modern Quantum Mechanics 2Nd Edition." Person New International edition (2014).
    Page 225, Eq. 3.8.49.

and:
    Schulten, Klaus, and Roy G. Gordon, Journal of Mathematical Physics 16.10 (1975): 1961-1970. Eq. 6, 9.
"""


import os

import torch
from joblib import Memory
from sympy import N
from sympy.physics.quantum.cg import CG

from torch_gauge import ROOT_DIR
from torch_gauge.o3 import O3Tensor, SphericalTensor
from torch_gauge.o3.wigner import csh_to_rsh

memory = Memory(os.path.join(ROOT_DIR, ".o3_cache"), verbose=0)


class LeviCivitaCoupler(torch.nn.Module):
    """
    Simple tensor coupling module when max_l==1.

    The input spherical tensors must have n_rep_dims==1 and aligned dimensions.
    """

    def __init__(self, metadata: torch.LongTensor):
        super().__init__()
        assert metadata.dim() == 1
        assert (
            len(metadata) == 2
        ), "Only SphericalTensor of max degree 1 is applicable for Cevi-Levita"
        self._metadata = metadata

    def forward(self, x1: SphericalTensor, x2: SphericalTensor, overlap_out=True):
        """
        Args:
            x1 (SphericalTensor): The first input ``SphericalTensor`` to be coupled.
            x2 (SphericalTensor): The second input ``SphericalTensor`` to be coupled.
            overlap_out (bool): If true, the coupling outputs from the same input feature index but
                different (l1, l2)s will be accumulated to the same representation index of the output SphericalTensor.

        Returns:
            A new SphericalTensor from Levi-Civita coupling.
        """
        assert x1.metadata.shape[0] == 1
        assert x2.metadata.shape[0] == 1
        assert x1.rep_dims[0] == x2.rep_dims[0]
        coupling_dim = x1.rep_dims[0]
        assert torch.all(x1.metadata[0].eq(self._metadata))
        assert torch.all(x2.metadata[0].eq(self._metadata))
        ten_l1_1 = x1.ten.narrow(
            coupling_dim, self._metadata[0], self._metadata[1] * 3
        ).unflatten(coupling_dim, (3, self._metadata[1]))
        ten_l1_2 = x2.ten.narrow(
            coupling_dim, self._metadata[0], self._metadata[1] * 3
        ).unflatten(coupling_dim, (3, self._metadata[1]))
        # 0x0->0
        out_000 = x1.ten.narrow(coupling_dim, 0, self._metadata[0]) * x2.ten.narrow(
            coupling_dim, 0, self._metadata[0]
        )
        # 0x1->1
        out_011 = (
            x1.ten.narrow(coupling_dim, 0, self._metadata[1])
            .unsqueeze(coupling_dim)
            .mul(ten_l1_2)
        )
        # 1x0->1
        out_101 = (
            x2.ten.narrow(coupling_dim, 0, self._metadata[1])
            .unsqueeze(coupling_dim)
            .mul(ten_l1_1)
        )
        # 1x1->0
        out_110 = (ten_l1_1 * ten_l1_2).sum(coupling_dim)
        # 1x1->1, note that cross works since (y,z,x) is a canonical order
        out_111 = torch.cross(ten_l1_1, ten_l1_2, dim=coupling_dim)
        if overlap_out:
            # Align and contract the coupling outputs
            out_l0 = out_000
            out_l0.narrow(coupling_dim, 0, self._metadata[1]).add(out_110)
            out_l1 = (
                out_111.add(out_101)
                .add(out_011)
                .flatten(coupling_dim, coupling_dim + 1)
            )
            return x1.self_like(torch.cat([out_l0, out_l1], dim=coupling_dim))
        else:
            # Concatenate the coupling outputs to form a augmented tensor
            out_l0 = torch.cat([out_000, out_110], dim=coupling_dim)
            out_l1 = torch.cat(
                [out_101, out_011, out_111], dim=coupling_dim + 1
            ).flatten(coupling_dim, coupling_dim + 1)
            return SphericalTensor(
                torch.cat([out_l0, out_l1], dim=coupling_dim),
                rep_dims=(coupling_dim,),
                metadata=torch.LongTensor(
                    [[self._metadata[0] + self._metadata[1], self._metadata[1] * 3]]
                ),
            )


def get_clebsch_gordan_coefficient(j1, j2, j, m1, m2, m):
    """
    Generate Clebsch-Gordan coefficients using sympy with caching.
    """
    # Matching the convention
    return float(N(CG(j1, m1, j2, m2, j, m).doit()))


# noinspection PyTypeChecker
@memory.cache
def get_rsh_cg_coefficients(j1, j2, j):
    csh_cg = torch.zeros(2 * j1 + 1, 2 * j2 + 1, 2 * j + 1, dtype=torch.double)
    for m1 in range(-j1, j1 + 1):
        for m2 in range(-j2, j2 + 1):
            if m1 + m2 < -j or m1 + m2 > j:
                continue
            csh_cg[j1 + m1, j2 + m2, j + m1 + m2] = get_clebsch_gordan_coefficient(
                j1, j2, j, m1, m2, m1 + m2
            )
    c2r_j1, c2r_j2, c2r_j = csh_to_rsh(j1), csh_to_rsh(j2), csh_to_rsh(j)
    # Adding a phase factor such that all coupling coefficients are real
    rsh_cg = torch.einsum(
        "abc,ai,bj,ck->ijk", csh_cg.to(torch.cdouble), c2r_j1, c2r_j2, c2r_j.conj()
    ) * (-1j) ** (j1 + j2 + j)
    assert torch.allclose(rsh_cg.imag, torch.zeros_like(csh_cg)), print(csh_cg, rsh_cg)
    return cg_compactify(rsh_cg.real, j1, j2, j)


def cg_compactify(coeffs, j1, j2, j):
    j1s = torch.arange(-j1, j1 + 1).view(2 * j1 + 1, 1, 1).expand_as(coeffs)
    j2s = torch.arange(-j2, j2 + 1).view(1, 2 * j2 + 1, 1).expand_as(coeffs)
    js = torch.arange(-j, j + 1).view(1, 1, 2 * j + 1).expand_as(coeffs)
    nonzero_mask = coeffs.abs() > 1e-12
    return torch.stack(
        [j1s[nonzero_mask], j2s[nonzero_mask], js[nonzero_mask], coeffs[nonzero_mask]],
        dim=0,
    )


class CGCoupler(torch.nn.Module):
    """
    General vectorized Clebsch-Gordan coupling module.

    Note:
        When the CGCoupler class is instantiated, a compact view of representation indices
        is generated for vectorizing Clebsch-Gordan coupling between two SphericalTensors. Depending
        on the setup of input SphericalTensors, this tabulating step can be time-consuming; the CGCoupler
        parameters should be saved when the user intends to use the model for inference.

    Attributes:
        metadata_in1 (torch.LongTensor): The metadata of the first input SphericalTensor to be coupled.
        metadata_in2 (torch.LongTensor): The metadata of the second input SphericalTensor to be coupled.
        metadata_out (torch.LongTensor): The metadata of the output SphericalTensor. Note that it depends
            on the coupling specifications, `overlap_out` and `trunc_in`.

    Args:
        metadata_1 (torch.LongTensor): The representation metadata of the first tensor to be coupled.
        metadata_2 (torch.LongTensor): The representation metadata of the second tensor to be coupled,
             must have the same length (number of l's) as ``metadata_1``.
        parity (int): The parity to be retained in coupling. 0: No parity selection; 1: Polar tensor; -1: Pseudo tensor.
        overlap_out (bool): If true, the coupling outputs from the same input feature index but different (l1, l2)s
             will be accumulated to the same representation index of the output SphericalTensor.
        trunc_in (bool): If true, the allowed feature indices (n) will be further truncated such that for
             each set of terms (l1, l2, n), the coupling results will saturate all possible (l_out, n) values
             of the output SphericalTensor.
        dtype (torch.dtype): The dtype for tensor to be passed in coupling, must be specified beforehand.

    Warning:
        When using overlap_out, the parity of the system may not be conserved.
    """

    def __init__(
        self,
        metadata_1: torch.LongTensor,
        metadata_2: torch.LongTensor,
        parity=0,
        overlap_out=True,
        trunc_in=True,
        dtype=torch.double,
    ):
        super().__init__()
        metadata_1 = torch.LongTensor(metadata_1)
        metadata_2 = torch.LongTensor(metadata_2)
        assert metadata_1.dim() == 1
        assert metadata_2.dim() == 1
        assert metadata_1.shape[0] == metadata_2.shape[0]
        self.metadata_out = None
        self.metadata_in1 = metadata_1
        self.metadata_in2 = metadata_2
        self.parity = parity
        self.dtype = dtype
        self._init_params(overlap_out, trunc_in)
        self.out_layout = torch.nn.Parameter(
            SphericalTensor.generate_rep_layout_1d_(self.metadata_out),
            requires_grad=False,
        )

    def _init_params(self, overlap_out, trunc_in):
        metadata_in = torch.stack([self.metadata_in1, self.metadata_in2], dim=0)
        max_n_out = torch.maximum(self.metadata_in1, self.metadata_in2)
        n_irreps_per_l = torch.arange(start=0, end=metadata_in.shape[1]) * 2 + 1
        repid_offsets_in = torch.cumsum(
            metadata_in * n_irreps_per_l.unsqueeze(0), dim=1
        )
        repid_offsets_in = torch.cat(
            [torch.LongTensor([[0], [0]]), repid_offsets_in[:, :-1]], dim=1
        ).long()
        cg_tilde, repids_in1, repids_in2, repids_out = [], [], [], []
        max_l = metadata_in.shape[1] - 1
        # Tabulate the output metadata and allowed coupling terms
        valid_coupling_ids = []
        metadata_out = torch.zeros_like(max_n_out)
        for lout in range(max_l + 1):
            for lin1 in range(max_l + 1):
                for lin2 in range(max_l + 1):
                    coupling_parity = (-1) ** (lout + lin1 + lin2)
                    if not self.parity == 0:
                        if self.parity != coupling_parity:
                            continue
                    if lin1 + lin2 < lout or abs(lin1 - lin2) > lout:
                        continue
                    if trunc_in:
                        if lin1 + lin2 > max_l:
                            continue
                        degeneracy = min(
                            metadata_in[0, lin1],
                            metadata_in[1, lin2],
                            max_n_out[lin1 + lin2],
                        )
                    else:
                        if lout > max_l:
                            continue
                        degeneracy = min(
                            metadata_in[0, lin1],
                            metadata_in[1, lin2],
                            max_n_out[lout],
                        )
                    if not overlap_out:
                        metadata_out[lout] += degeneracy
                    elif degeneracy > metadata_out[lout]:
                        metadata_out[lout] = degeneracy
                    if degeneracy > 0:
                        valid_coupling_ids.append((lout, lin1, lin2, degeneracy))

        repid_offsets_out = torch.cumsum(metadata_out * n_irreps_per_l, dim=0)
        repid_offsets_out = torch.cat(
            [torch.LongTensor([0]), repid_offsets_out[:-1]], dim=0
        )

        out_ns_offset, lout_last = 0, 0
        # Generate flattened coupling coefficients
        for (lout, lin1, lin2, degeneracy) in valid_coupling_ids:
            if lout > lout_last:
                out_ns_offset = 0
            cg_source = get_rsh_cg_coefficients(lin1, lin2, lout)
            cg_segment = cg_source.repeat_interleave(degeneracy, dim=1)
            ns_segment = torch.arange(degeneracy).repeat(cg_source.shape[1])
            # Calculating the representation IDs for the coupling tensors
            repids_in1_3j = (
                repid_offsets_in[0, lin1]
                + (cg_segment[0] + lin1) * metadata_in[0, lin1]
                + ns_segment
            )
            repids_in2_3j = (
                repid_offsets_in[1, lin2]
                + (cg_segment[1] + lin2) * metadata_in[1, lin2]
                + ns_segment
            )
            repids_out_3j = (
                repid_offsets_out[lout]
                + (cg_segment[2] + lout) * metadata_out[lout]
                + out_ns_offset
                + ns_segment
            )

            cg_tilde.append(cg_segment[3])
            repids_in1.append(repids_in1_3j)
            repids_in2.append(repids_in2_3j)
            repids_out.append(repids_out_3j)
            if not overlap_out:
                out_ns_offset += degeneracy
            lout_last = lout

        self.cg_tilde = torch.nn.Parameter(
            torch.cat(cg_tilde).type(self.dtype), requires_grad=False
        )
        self.repids_in1 = torch.nn.Parameter(
            torch.cat(repids_in1).long(), requires_grad=False
        )
        self.repids_in2 = torch.nn.Parameter(
            torch.cat(repids_in2).long(), requires_grad=False
        )
        self.repids_out = torch.nn.Parameter(
            torch.cat(repids_out).long(), requires_grad=False
        )
        # Do not transfer metadata to device
        self.metadata_out = metadata_out

    def forward(self, x1: SphericalTensor, x2: SphericalTensor) -> SphericalTensor:
        """
        Args:
            x1 (SphericalTensor): The first input ``SphericalTensor`` to be coupled,
                must have exactly 1 representation dimension.
            x2 (SphericalTensor): The second input ``SphericalTensor`` to be coupled,
                must have exactly 1 representation dimension.

        Returns:
            A new SphericalTensor with ``self.metadata_out`` from C-G coupling.
        """
        assert len(x1.rep_dims) == 1
        assert len(x2.rep_dims) == 1
        assert x1.rep_dims[0] == x2.rep_dims[0]
        coupling_dim = x1.rep_dims[0]
        assert torch.all(x1.metadata[0].eq(self.metadata_in1))
        assert torch.all(x2.metadata[0].eq(self.metadata_in2))
        x1_tilde = torch.index_select(x1.ten, dim=coupling_dim, index=self.repids_in1)
        x2_tilde = torch.index_select(x2.ten, dim=coupling_dim, index=self.repids_in2)
        broadcast_shape = tuple(
            self.cg_tilde.shape[0] if d == coupling_dim else 1
            for d in range(x1.ten.dim())
        )
        out_tilde = x1_tilde * x2_tilde * self.cg_tilde.view(broadcast_shape)
        out_shape = tuple(
            self.out_layout.shape[1] if d == coupling_dim else x1.ten.shape[d]
            for d in range(x1.ten.dim())
        )
        out_ten = torch.zeros(
            out_shape, dtype=x1_tilde.dtype, device=x1_tilde.device
        ).index_add_(
            coupling_dim,
            self.repids_out,
            out_tilde,
        )
        return SphericalTensor(
            out_ten,
            rep_dims=(coupling_dim,),
            metadata=self.metadata_out.unsqueeze(0),
            rep_layout=(self.out_layout,),
        )


class CGPCoupler(torch.nn.Module):
    """
    Parity-aware vectorized Clebsch-Gordan coupling module.

    Note:
        When the CGPCoupler class is instantiated, a compact view of representation indices
        is generated for vectorizing Clebsch-Gordan coupling between two O3Tensors. Depending
        on the setup of input O3Tensors, this tabulating step can be time-consuming; the CGPCoupler
        parameters should be saved when the user intends to use the model for inference.

    Attributes:
        metadata_in1 (torch.LongTensor): The metadata of the first input O3Tensor to be coupled.
        metadata_in2 (torch.LongTensor): The metadata of the second input O3Tensor to be coupled.
        metadata_out (torch.LongTensor): The metadata of the output O3Tensor. Note that it depends
            on the coupling specifications, `overlap_out` and `trunc_in`.

    Args:
        metadata_1 (torch.LongTensor): The representation metadata of the first tensor to be coupled.
        metadata_2 (torch.LongTensor): The representation metadata of the second tensor to be coupled,
             must have the same length (number of ls and ps) as ``metadata_1``.
        trunc_in (bool): If true, the allowed feature indices (n) will be further truncated such that for
             each set of terms (l1, l2, n), the coupling results will saturate all possible (l_out, n) values
             of the output O3Tensor.
        dtype (torch.dtype): The dtype for tensor to be passed in coupling, must be specified beforehand.

    """

    def __init__(
        self,
        metadata_1: torch.LongTensor,
        metadata_2: torch.LongTensor,
        trunc_in=True,
        dtype=torch.double,
    ):
        super().__init__()
        metadata_1 = torch.LongTensor(metadata_1)
        metadata_2 = torch.LongTensor(metadata_2)
        assert metadata_1.dim() == 1
        assert metadata_2.dim() == 1
        assert metadata_1.shape[0] == metadata_2.shape[0]
        assert metadata_1.shape[0] % 2 == 0
        self.metadata_out = None
        self.metadata_in1 = metadata_1
        self.metadata_in2 = metadata_2
        self.dtype = dtype
        self._init_params(trunc_in)
        self.out_layout = torch.nn.Parameter(
            O3Tensor.generate_rep_layout_1d_(self.metadata_out), requires_grad=False
        )

    def _init_params(self, trunc_in):
        metadata_in = torch.stack([self.metadata_in1, self.metadata_in2], dim=0)
        max_n_out = torch.maximum(self.metadata_in1, self.metadata_in2)
        n_irreps_per_l = torch.arange(start=0, end=metadata_in.shape[1] // 2) * 2 + 1
        n_irreps_per_lp = n_irreps_per_l.repeat_interleave(2)
        repid_offsets_in = torch.cumsum(
            metadata_in * n_irreps_per_lp.unsqueeze(0), dim=1
        )
        repid_offsets_in = torch.cat(
            [torch.LongTensor([[0], [0]]), repid_offsets_in[:, :-1]], dim=1
        ).long()
        cg_tilde, repids_in1, repids_in2, repids_out = [], [], [], []
        max_l = metadata_in.shape[1] // 2 - 1
        # Tabulate the output metadata and allowed coupling terms
        valid_coupling_ids = []
        metadata_out = torch.zeros_like(max_n_out)
        for lout in range(max_l + 1):
            for pout in (1, -1):
                for lin1 in range(max_l + 1):
                    for lin2 in range(max_l + 1):
                        for pin1 in (1, -1):
                            for pin2 in (1, -1):
                                coupling_parity = (-1) ** (lout + lin1 + lin2)
                                # parity selection rule
                                if pin1 * pin2 * coupling_parity != pout:
                                    continue
                                # Angular selection rule
                                if lin1 + lin2 < lout or abs(lin1 - lin2) > lout:
                                    continue
                                lpin1 = 2 * lin1 + (1 - pin1) // 2
                                lpin2 = 2 * lin2 + (1 - pin2) // 2
                                lpout = 2 * lout + (1 - pout) // 2

                                if trunc_in:
                                    if lin1 + lin2 > max_l:
                                        continue
                                    degeneracy = min(
                                        metadata_in[0, lpin1],
                                        metadata_in[1, lpin2],
                                        max_n_out[2 * (lin1 + lin2) + (1 - pout) // 2],
                                    )
                                else:
                                    if lout > max_l:
                                        continue
                                    degeneracy = min(
                                        metadata_in[0, lpin1],
                                        metadata_in[1, lpin2],
                                        max_n_out[lpout],
                                    )
                                metadata_out[lpout] += degeneracy
                                if degeneracy > 0:
                                    valid_coupling_ids.append(
                                        (lpout, lpin1, lpin2, degeneracy)
                                    )

        repid_offsets_out = torch.cumsum(metadata_out * n_irreps_per_lp, dim=0)
        repid_offsets_out = torch.cat(
            [torch.LongTensor([0]), repid_offsets_out[:-1]], dim=0
        )

        out_ns_offset, lpout_last = 0, 0
        # Generate flattened coupling coefficients
        for (lpout, lpin1, lpin2, degeneracy) in valid_coupling_ids:
            if lpout > lpout_last:
                out_ns_offset = 0
            lin1, lin2, lout = lpin1 // 2, lpin2 // 2, lpout // 2
            cg_source = get_rsh_cg_coefficients(lin1, lin2, lout)
            cg_segment = cg_source.repeat_interleave(degeneracy, dim=1)
            ns_segment = torch.arange(degeneracy).repeat(cg_source.shape[1])
            # Calculating the representation IDs for the coupling tensors
            repids_in1_3j = (
                repid_offsets_in[0, lpin1]
                + (cg_segment[0] + lin1) * metadata_in[0, lpin1]
                + ns_segment
            )
            repids_in2_3j = (
                repid_offsets_in[1, lpin2]
                + (cg_segment[1] + lin2) * metadata_in[1, lpin2]
                + ns_segment
            )
            repids_out_3j = (
                repid_offsets_out[lpout]
                + (cg_segment[2] + lout) * metadata_out[lpout]
                + out_ns_offset
                + ns_segment
            )

            cg_tilde.append(cg_segment[3])
            repids_in1.append(repids_in1_3j)
            repids_in2.append(repids_in2_3j)
            repids_out.append(repids_out_3j)
            out_ns_offset += degeneracy
            lpout_last = lpout

        self.cg_tilde = torch.nn.Parameter(
            torch.cat(cg_tilde).type(self.dtype), requires_grad=False
        )
        self.repids_in1 = torch.nn.Parameter(
            torch.cat(repids_in1).long(), requires_grad=False
        )
        self.repids_in2 = torch.nn.Parameter(
            torch.cat(repids_in2).long(), requires_grad=False
        )
        self.repids_out = torch.nn.Parameter(
            torch.cat(repids_out).long(), requires_grad=False
        )
        # Do not transfer metadata to device
        self.metadata_out = metadata_out

    def forward(self, x1: O3Tensor, x2: O3Tensor) -> O3Tensor:
        """
        Args:
            x1 (O3Tensor): The first input ``O3Tensor`` to be coupled,
                must have exactly 1 representation dimension.
            x2 (O3Tensor): The second input ``O3Tensor`` to be coupled,
                must have exactly 1 representation dimension.

        Returns:
            A new O3Tensor with ``self.metadata_out`` from C-G coupling.
        """
        assert len(x1.rep_dims) == 1
        assert len(x2.rep_dims) == 1
        assert x1.rep_dims[0] == x2.rep_dims[0]
        coupling_dim = x1.rep_dims[0]
        assert torch.all(x1.metadata[0].eq(self.metadata_in1))
        assert torch.all(x2.metadata[0].eq(self.metadata_in2))
        x1_tilde = torch.index_select(x1.ten, dim=coupling_dim, index=self.repids_in1)
        x2_tilde = torch.index_select(x2.ten, dim=coupling_dim, index=self.repids_in2)
        broadcast_shape = tuple(
            self.cg_tilde.shape[0] if d == coupling_dim else 1
            for d in range(x1.ten.dim())
        )
        out_tilde = x1_tilde * x2_tilde * self.cg_tilde.view(broadcast_shape)
        out_shape = tuple(
            self.out_layout.shape[1] if d == coupling_dim else x1.ten.shape[d]
            for d in range(x1.ten.dim())
        )
        out_ten = torch.zeros(
            out_shape, dtype=x1_tilde.dtype, device=x1_tilde.device
        ).index_add_(
            coupling_dim,
            self.repids_out,
            out_tilde,
        )
        return O3Tensor(
            out_ten,
            rep_dims=(coupling_dim,),
            metadata=self.metadata_out.unsqueeze(0),
            rep_layout=(self.out_layout.data,),
        )
