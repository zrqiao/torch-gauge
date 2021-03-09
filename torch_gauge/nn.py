import math

import torch
import torch.nn.functional as F
from deprecated import deprecated

from torch_gauge.o3 import O3Tensor, SphericalTensor


class SSP(torch.nn.Softplus):
    """Shifted SoftPlus activation function."""

    def __init__(self, beta=1, threshold=20):
        super().__init__(beta, threshold)

    def forward(self, input):
        """"""
        return F.softplus(input, self.beta, self.threshold) - math.sqrt(2)


class Swish_fn(torch.autograd.Function):
    """Swish activation function."""

    @staticmethod
    def forward(ctx, i):
        """"""
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """"""
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(torch.nn.Module):
    def forward(self, input_tensor):
        """"""
        return Swish_fn.apply(input_tensor)


class IELin(torch.nn.Module):
    r"""
    Irrep-wise Equivariant Linear Layer.

    This module takes a spherical tensor and perform linear transformation within
    the feature channels spanned by each irreducible representation index, (l, m):

    .. math::

        \mathbf{h}^{\mathrm{out}}_{l,m} = \mathbf{W}^{l} \cdot \mathbf{h}^{\mathrm{in}}_{l,m}

    The matrix multiplication is always performed on the last dimension of the spherical tensors.

    Note:
        This function is vectorized by tensor padding. An additional memory cost might
        arise, and it's recommended to used benchmark=True to minimize this overhead.

    Attributes:
        metadata_in (torch.LongTensor): the number of non-degenerate feature channels of the input tensor
            for each l (and the associated m(s)).
        metadata_out (torch.LongTensor): the number of non-degenerate feature channels of the output tensor
            for each l (and the associated m(s)).
        group (str): The group index of the tensors to be passed.
        benchmark (bool): search the optimal padding size when creating the module.
    """

    def __init__(self, metadata_in, metadata_out, group="o3", benchmark=True):
        super().__init__()
        assert metadata_in.dim() == 1
        assert len(metadata_in) == len(metadata_out)
        self._metadata_in = metadata_in
        self._metadata_out = metadata_out
        group = group.lower()
        if group == "so3":
            self.tensor_class = SphericalTensor
            self.n_irreps_per_l = torch.arange(start=0, end=metadata_in.size(0)) * 2 + 1
        elif group == "o3":
            self.tensor_class = O3Tensor
            n_irreps_per_l = torch.arange(start=0, end=metadata_in.size(0) // 2) * 2 + 1
            self.n_irreps_per_l = n_irreps_per_l.repeat_interleave(2)
        else:
            raise NotImplementedError(f"The group {group} is not supported in IELin")

        padding_size_in = metadata_in.max().item()
        padding_size_out = metadata_out.max().item()

        # TODO: estimate the optimal padding size

        # Padding rules:
        # I - non-degenerate matrices, degenerate vectors, cat reduce
        # J - non-degenerate matrices, non-degenerate vectors, sum reduce
        # m - degenerate matrices, non-degenerate vectors, cat reduce
        # Block index order: m->I->J->padding(k)
        vector_padding_idx = []
        vector_padding_mask = []
        src_offset = 0
        for lp_idx, nin_lpm in enumerate(metadata_in):
            nin_lpm = nin_lpm.item()
            nout_lpm = metadata_out[lp_idx].item()
            if metadata_in[lp_idx] == 0:
                continue
            bmv_degeneracy_lpm = -(nout_lpm // (-padding_size_out))
            if bmv_degeneracy_lpm == 0:
                src_offset += nin_lpm * self.n_irreps_per_l[lp_idx]
                continue
            for m in range(self.n_irreps_per_l[lp_idx]):
                # J-padding
                vector_padding_idx_lpm = torch.zeros(
                    -(nin_lpm // (-padding_size_in)) * padding_size_in, dtype=torch.long
                )
                vector_padding_mask_lpm = torch.zeros(
                    -(nin_lpm // (-padding_size_in)) * padding_size_in, dtype=torch.long
                )
                vector_padding_idx_lpm[:nin_lpm] = (
                    torch.arange(nin_lpm, dtype=torch.long) + src_offset
                )
                vector_padding_mask_lpm[:nin_lpm] = 1
                # I-padding
                vector_padding_idx.append(
                    vector_padding_idx_lpm.repeat(bmv_degeneracy_lpm)
                )
                vector_padding_mask.append(
                    vector_padding_mask_lpm.repeat(bmv_degeneracy_lpm)
                )
                src_offset += nin_lpm
        self.vector_padding_idx = torch.nn.Parameter(
            torch.cat(vector_padding_idx), requires_grad=False
        )
        self.vector_padding_mask = torch.nn.Parameter(
            torch.cat(vector_padding_mask).bool(), requires_grad=False
        )

        matrix_select_idx = []
        mat_offset = 0
        for lp_idx, nin_lpm in enumerate(metadata_in):
            nin_lpm = nin_lpm.item()
            nout_lpm = metadata_out[lp_idx].item()
            nblocks_lp = (-(nin_lpm // (-padding_size_in))) * (
                -(nout_lpm // (-padding_size_out))
            )
            if nblocks_lp > 0:
                matrix_select_idx.append(
                    (torch.arange(nblocks_lp, dtype=torch.long) + mat_offset).repeat(
                        self.n_irreps_per_l[lp_idx]
                    )
                )
                mat_offset += nblocks_lp
        self.matrix_select_idx = torch.nn.Parameter(
            torch.cat(matrix_select_idx), requires_grad=False
        )
        self.n_mats = mat_offset
        self.n_gathered_mats = self.matrix_select_idx.shape[0]

        out_reduce_idx = []
        out_reduce_mask = []
        dst_offset = 0
        for lp_idx, nout_lpm in enumerate(metadata_out):
            nin_lpm = metadata_in[lp_idx].item()
            nout_lpm = nout_lpm.item()
            if nout_lpm == 0:
                continue
            bmv_degeneracy_lpm = -(nout_lpm // (-padding_size_out))
            in_j_degeneracy_lpm = -(nin_lpm // (-padding_size_in))
            if in_j_degeneracy_lpm == 0:
                dst_offset += nout_lpm * self.n_irreps_per_l[lp_idx]
                continue
            for m in range(self.n_irreps_per_l[lp_idx]):
                dst_idx_lpm = (
                    (
                        dst_offset
                        + torch.arange(
                            padding_size_out * bmv_degeneracy_lpm, dtype=torch.long
                        )
                    )
                    .view(bmv_degeneracy_lpm, 1, padding_size_out)
                    .expand(bmv_degeneracy_lpm, in_j_degeneracy_lpm, padding_size_out)
                    .view(-1)
                )
                dst_mask_lpm = torch.zeros(
                    padding_size_out * bmv_degeneracy_lpm, dtype=torch.long
                )
                dst_mask_lpm[:nout_lpm] = 1
                dst_mask_lpm = (
                    dst_mask_lpm.view(bmv_degeneracy_lpm, 1, padding_size_out)
                    .expand(bmv_degeneracy_lpm, in_j_degeneracy_lpm, padding_size_out)
                    .view(-1)
                )
                out_reduce_idx.append(dst_idx_lpm)
                out_reduce_mask.append(dst_mask_lpm)
                dst_offset += nout_lpm
        self.out_reduce_mask = torch.nn.Parameter(
            torch.cat(out_reduce_mask).bool(), requires_grad=False
        )
        self.out_reduce_idx = torch.nn.Parameter(
            torch.cat(out_reduce_idx)[self.out_reduce_mask], requires_grad=False
        )

        self.linears = torch.nn.Parameter(
            torch.rand(self.n_mats, padding_size_in, padding_size_out) * 0.1
        )
        self.padding_size_in = padding_size_in
        self.padding_size_out = padding_size_out

        self.out_layout = torch.nn.Parameter(
            self.tensor_class.generate_rep_layout_1d_(self._metadata_out),
            requires_grad=False,
        )
        self.num_out_channels = torch.sum(self._metadata_out).item()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # TODO: fix the fan_in calculations
        torch.nn.init.kaiming_uniform_(self.linears, a=math.sqrt(5), mode="fan_in")

    def forward(self, x: SphericalTensor) -> SphericalTensor:
        """
        Args:
            x: The input SphericalTensor.

        Returns:
            The data tensor of the output spherical tensor.
        """
        assert x.rep_dims[-1] == x.ten.dim() - 1
        assert torch.all(x.metadata[-1].eq(self._metadata_in)), (
            f"Expected the SphericalTensor x and self._metadata_in to have the "
            f"same irrep metadata along the last dimension, got {x.metadata[-1]}"
            f" and {self._metadata_in} instead"
        )
        in_ten = x.ten.view(-1, x.ten.shape[-1])
        padded_in_ten = torch.index_select(
            in_ten, dim=1, index=self.vector_padding_idx
        ).mul_(self.vector_padding_mask)
        # to (lIJ)-B-k layout
        padded_in_ten = padded_in_ten.view(
            in_ten.shape[0], self.n_gathered_mats, self.padding_size_in
        ).transpose(0, 1)
        gathered_linears = torch.index_select(
            self.linears, dim=0, index=self.matrix_select_idx
        )
        padded_out_ten = (
            torch.bmm(padded_in_ten, gathered_linears)
            .transpose(0, 1)
            .contiguous()
            .view(in_ten.shape[0], -1)
        )
        out_ten = torch.zeros(
            in_ten.shape[0],
            self.out_layout.shape[1],
            dtype=in_ten.dtype,
            device=in_ten.device,
        ).index_add_(1, self.out_reduce_idx, padded_out_ten[:, self.out_reduce_mask])
        out_ten = out_ten.view(*x.ten.shape[:-1], -1)

        out_metadata = x.metadata.clone()
        out_metadata[-1] = self._metadata_out
        out_rep_layout = x.rep_layout[:-1] + (self.out_layout.data,)
        return self.tensor_class(
            out_ten,
            rep_dims=x.rep_dims,
            metadata=out_metadata,
            rep_layout=out_rep_layout,
            num_channels=x.num_channels[:-1] + (self.num_out_channels,),
        )


class IELinSerial(torch.nn.Module):
    r"""
    Irrep-wise Equivariant Linear Layer.

    This module takes a spherical tensor and perform linear transformation within
    the feature channels spanned by each irreducible representation index, (l, m):

    .. math::

        \mathbf{h}^{\mathrm{out}}_{l,m} = \mathbf{W}^{l} \cdot \mathbf{h}^{\mathrm{in}}_{l,m}

    The matrix multiplication is always performed on the last dimension of the spherical tensors.

    Note:
        The implementation of this operation is not optimized.

    Attributes:
        metadata_in (torch.LongTensor): the number of non-degenerate feature channels of the input tensor
            for each l (and the associated m(s)).
        metadata_out (torch.LongTensor): the number of non-degenerate feature channels of the output tensor
            for each l (and the associated m(s)).
        group (str): The group index of the tensors to be passed.
    """

    @deprecated(version="1.4.0", reason="Please use the vectorized LELin module")
    def __init__(self, metadata_in, metadata_out, group="o3"):
        super().__init__()
        assert metadata_in.dim() == 1
        assert len(metadata_in) == len(metadata_out)
        self._metadata_in = metadata_in
        self._metadata_out = metadata_out
        group = group.lower()
        if group == "so3":
            self.tensor_class = SphericalTensor
            self.n_irreps_per_l = torch.arange(start=0, end=metadata_in.size(0)) * 2 + 1
        elif group == "o3":
            self.tensor_class = O3Tensor
            n_irreps_per_l = torch.arange(start=0, end=metadata_in.size(0) // 2) * 2 + 1
            self.n_irreps_per_l = n_irreps_per_l.repeat_interleave(2)
        else:
            raise NotImplementedError(f"The group {group} is not supported in IELin")
        # The bias must be False
        self.linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    metadata_in[l],
                    metadata_out[l],
                    bias=False,
                )
                if self._metadata_out[l] > 0 and self._metadata_in[l] > 0
                else None
                for l, _ in enumerate(metadata_in)
            ]
        )
        self._end_inds_in = torch.cumsum(self._metadata_in * self.n_irreps_per_l, dim=0)
        self._start_inds_in = torch.cat([torch.LongTensor([0]), self._end_inds_in[:-1]])
        self.out_layout = torch.nn.Parameter(
            self.tensor_class.generate_rep_layout_1d_(self._metadata_out),
            requires_grad=False,
        )
        self.num_out_channels = torch.sum(self._metadata_out).item()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for linear in self.linears:
            if linear is None:
                continue
            torch.nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5), mode="fan_in")

    def forward(self, x: SphericalTensor) -> SphericalTensor:
        """
        Args:
            x: The input SphericalTensor.

        Returns:
            The data tensor of the output spherical tensor.
        """
        assert x.rep_dims[-1] == x.ten.dim() - 1
        assert torch.all(x.metadata[-1].eq(self._metadata_in)), (
            f"Expected the SphericalTensor x and self._metadata_in to have the "
            f"same irrep metadata along the last dimension, got {x.metadata[-1]}"
            f" and {self._metadata_in} instead"
        )
        outs = []
        for l, linear_l in enumerate(self.linears):
            if linear_l is None:
                if self._metadata_out[l] > 0:
                    outs.append(
                        torch.zeros(
                            *x.shape[:-1],
                            self._metadata_out[l] * self.n_irreps_per_l[l],
                            dtype=x.ten.dtype,
                            device=x.ten.device,
                        )
                    )
                continue
            in_l = x.ten[
                ..., self._start_inds_in[l] : self._end_inds_in[l]
            ].contiguous()
            out_l = linear_l(
                in_l.unflatten(-1, (self.n_irreps_per_l[l], self._metadata_in[l]))
            )
            outs.append(
                out_l.view(
                    *x.shape[:-1], self._metadata_out[l] * self.n_irreps_per_l[l]
                )
            )

        out_ten = torch.cat(outs, dim=-1)
        out_metadata = x.metadata.clone()
        out_metadata[-1] = self._metadata_out
        out_rep_layout = x.rep_layout[:-1] + (self.out_layout.data,)
        return self.tensor_class(
            out_ten,
            rep_dims=x.rep_dims,
            metadata=out_metadata,
            rep_layout=out_rep_layout,
            num_channels=x.num_channels[:-1] + (self.num_out_channels,),
        )


class RepNorm1d(torch.nn.Module):
    r"""
    The (experimental) Representation Normalization layer.

    .. math::

        \mathrm{RepNorm}(\mathbf{h})_{l,m} = \mathrm{BatchNorm}(||\mathbf{h}_l||)
        \cdot \\frac{\mathbf{h}_{l}}{ |(1-{\\beta})||\mathbf{h}_{l}|| + {\\beta}| + \\epsilon}

    Heuristically, the trainable :math:`\mathbf{\\beta}` controls the fraction of norm
    information to be retained.

    The specified invariant channels are not affected by the gauge factorization.
    """

    def __init__(
        self, num_channels, norm="batch", momentum=0.1, eps=1e-2, n_invariant_channels=0
    ):
        super().__init__()
        self._num_channels = num_channels
        self._n_invariant_channels = n_invariant_channels
        self._eps = eps
        if norm == "batch":
            self.norm = torch.nn.BatchNorm1d(
                num_features=num_channels, momentum=momentum, affine=False
            )
        elif norm == "node":
            self.norm = torch.nn.LayerNorm(
                normalized_shape=num_channels, elementwise_affine=False
            )
        else:
            raise NotImplementedError
        # TODO: initialization schemes
        self.beta = torch.nn.Parameter(
            torch.rand(self._num_channels - self._n_invariant_channels)
        )

    def forward(self, x: SphericalTensor) -> (torch.Tensor, SphericalTensor):
        """
        Args:
            x: The input SphericalTensor.

        Returns:
            (tuple): tuple containing:
                (torch.Tensor): The normalized invariant content.

                (SphericalTensor): The "pure gauge" spherical tensor.
        """
        x0 = x.invariant()
        assert x0.dim() == 2
        assert self._n_invariant_channels <= x.metadata[0][0]
        x1 = self.norm(x0)
        divisor = (
            torch.abs(
                x0[:, self._n_invariant_channels :].mul(1 - self.beta).add(self.beta)
            )
            + self._eps
        )
        divisor_broadcasted = torch.index_select(
            divisor,
            dim=1,
            index=(
                x.rep_layout[0][2, self._n_invariant_channels :]
                - self._n_invariant_channels
            ),
        )
        x2_ten = torch.cat(
            [
                torch.ones_like(x1[:, : self._n_invariant_channels]),
                x.ten[:, self._n_invariant_channels :].div(divisor_broadcasted),
            ],
            dim=1,
        )
        x2 = x.self_like(x2_ten)
        # x2.ten = x2.ten.div((x.rep_layout[0][0] * 2 + 1).type(x0.dtype).sqrt().unsqueeze(0))
        return x1, x2
