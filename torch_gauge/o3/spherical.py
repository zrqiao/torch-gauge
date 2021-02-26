from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from torch_gauge.o3.functional import NormContraction1d, NormContraction2d


@dataclass
class SphericalTensor:
    """
    The SphericalTensor class tracks the SO(3) representation indices
    of a flattened data tensor for efficient group algebra operations.

    All angular and magnetic quantum number indices are treated as equivalent,
    and all non-degenerate feature channels are regarded as the "principal"
    indices, n. Note that the angular quantum number indices (m) for each l
    range from -l to l as in the physicist's convention.

    Two indexing tensors are maintained, ``metadata`` and ``rep_layout``.

    Attributes:
        data_ten (torch.Tensor): The tensor storing auto-differentiable data.
        metadata (torch.LongTensor): The minimal information which specifies
            the number of non-degenerate feature channels for each dimension
            and each l (and the associated m(s)).
        rep_layout (Tuple[torch.LongTensor]): Stores the pointer (l[i], m[i], n[i])
            for each index i along the SO(3) representation dimension of the
            flattened data tensor.

    Args:
        data_ten (torch.Tensor): the underlying flattened data tensor in the
             strictly incremental [l,m,n] order, for which the rep_dim has N elements
        rep_dims (tuple of ints): the spherical tensor representation dimension(s). When
             n_rep_dims > 1, must be contiguous dimensions.
        metadata (torch.LongTensor): (n_rep_dims x n_l) specification of the number of unique
             channels (n) for each angular momentum index l.
        rep_layout (tuple of ints, optional): the (3xN) index tensor for (l[i],m[i],n[i]).
        num_channels (tuple of ints, optional): the number of unique channels per rep_dim.
    """

    def __init__(
        self,
        data_ten: torch.Tensor,
        rep_dims: Tuple[int, ...],
        metadata: torch.LongTensor,
        rep_layout: Optional[Tuple[torch.LongTensor, ...]] = None,
        num_channels: Optional[Tuple[int, ...]] = None,
    ):
        self.ten = data_ten
        self.metadata = metadata
        self.rep_dims = rep_dims
        assert self.rep_dims[-1] < self.ten.dim()
        self._norm_eps = 1e-4
        if rep_layout:
            self.rep_layout = rep_layout
        else:
            if len(rep_dims) > 1:
                assert rep_dims[1] - rep_dims[0] == 1
            self.rep_layout = self.generate_rep_layout()
        if num_channels:
            self.num_channels = num_channels
        else:
            self.num_channels = tuple(torch.sum(self.metadata, dim=1).long().tolist())

    @property
    def shape(self):
        return self.ten.shape

    @property
    def device(self):
        return self.ten.device

    def self_like(self, new_data_ten: torch.Tensor):
        """
        Args:
            new_data_ten: The underlying data of the new SphericalTensor.

        Returns:
             A SphericalTensor with identical layout but new data tensor.
        """
        return SphericalTensor(
            new_data_ten,
            rep_dims=self.rep_dims,
            metadata=self.metadata,
            rep_layout=self.rep_layout,
            num_channels=self.num_channels,
        )

    def mul_(self, other: "SphericalTensor"):
        # Be careful that this operation will break equivariance
        self.ten.mul_(other.ten)
        return self.ten

    def add_(self, other: "SphericalTensor"):
        self.ten.add_(other.ten)
        return self.ten

    def __mul__(self, other: "SphericalTensor"):
        return self.self_like(self.ten * other.ten)

    def __add__(self, other: "SphericalTensor"):
        return self.self_like(self.ten + other.ten)

    def __sub__(self, other: "SphericalTensor"):
        return self.self_like(self.ten - other.ten)

    def scalar_mul(self, other: torch.Tensor, inplace=False):
        """
        Performing representation-wise scalar multiplication.

        Args:
            other (torch.Tensor): along rep_dims, the shape must be the number
                of unique channels (n); along all other dimensions the shape must
                be the same as the data tensor.
            inplace (bool): If true, self.ten is updated in-place.
        """
        if len(self.rep_dims) == 1:
            broadcasted_other = torch.index_select(
                other, dim=self.rep_dims[0], index=self.rep_layout[0][2, :]
            )
        elif len(self.rep_dims) == 2:
            broadcasted_other = torch.index_select(
                other.view(
                    *other.shape[: self.rep_dims[0]],
                    self.num_channels[0] * self.num_channels[1],
                    *other.shape[self.rep_dims[1] + 1 :],
                ),
                dim=self.rep_dims[0],
                index=(
                    self.rep_layout[0][2, :].unsqueeze(1) * self.num_channels[1]
                    + self.rep_layout[1][2, :].unsqueeze(0)
                ).view(-1),
            ).view_as(self.ten)
        else:
            raise NotImplementedError

        if inplace:
            self.ten.mul_(broadcasted_other)
            return self
        else:
            return self.self_like(self.ten * broadcasted_other)

    def dot(self, other: "SphericalTensor", dim: int):
        """
        Performing inner product along a representation dimension.

        If self.n_rep_dim==1, a torch.Tensor is returned;
        if self.n_rep_dim==2, a SphericalTensor with n_rep_dim==1 is returned.

        Args:
            other (SphericalTensor): Must be a 1-d spherical tensor
                with the same number of dimensions and broadcastable to self.ten.
                When self.n_rep_dim==2, the kept dimension in self.rep_dims
                of other must be singleton.
            dim (int): The dimension to perform the inner product.
        """
        dotdim_idx = self.rep_dims.index(dim)
        assert other.rep_dims[0] == dim
        assert torch.all(self.metadata[dotdim_idx].eq(other.metadata[0]))
        out_ten = self.ten.mul(other.ten).sum(dim)
        if len(self.rep_dims) == 1:
            return out_ten
        elif len(self.rep_dims) == 2:
            dimid_kept = 1 - dotdim_idx
            assert other.ten.shape[self.rep_dims[dimid_kept]] == 1
            return self.__class__(
                out_ten,
                rep_dims=(self.rep_dims[dimid_kept],),
                metadata=self.metadata[dimid_kept].unsqueeze(0),
                rep_layout=(self.rep_layout[dimid_kept],),
                num_channels=(self.num_channels[dimid_kept],),
            )
        else:
            raise NotImplementedError

    def rep_dot(self, other: "SphericalTensor", dim: int):
        """
        Performing channel-wise inner products.

        If self.n_rep_dim==1, a torch.Tensor is returned;
        if self.n_rep_dim==2, a SphericalTensor with n_rep_dim==1 is returned.

        Args:
            other (SphericalTensor): must be a 1-d spherical tensor
                with the same number of dimensions and broadcastable to self.ten.
            dim (int): the dimension to perform the channel-wise inner product.
        """
        dotdim_idx = self.rep_dims.index(dim)
        assert other.rep_dims[0] == dim
        assert torch.all(self.metadata[dotdim_idx].eq(other.metadata[0]))
        mul_ten = self.ten * other.ten
        out_shape = list(mul_ten.shape)
        out_shape[dim] = self.num_channels[dotdim_idx]
        out_ten = torch.zeros(
            out_shape, device=mul_ten.device, dtype=mul_ten.dtype
        ).index_add_(
            dim,
            self.rep_layout[dotdim_idx][2, :],
            mul_ten,
        )
        # Note that the argument names of index_add in pytorch is changing from tensor to source
        # in 1.7-1.8. Not using keywords for compatibility
        if len(self.rep_dims) == 1:
            return out_ten
        elif len(self.rep_dims) == 2:
            # A little trick here
            dimid_kept = 1 - dotdim_idx
            return self.__class__(
                out_ten,
                rep_dims=(self.rep_dims[dimid_kept],),
                metadata=self.metadata[dimid_kept].unsqueeze(0),
                rep_layout=(self.rep_layout[dimid_kept],),
                num_channels=(self.num_channels[dimid_kept],),
            )
        else:
            raise NotImplementedError

    def rep_outer(self, other: "SphericalTensor") -> "SphericalTensor":
        """
        Returns the batched outer product of two 1-d spherical tensors.

        The rep_dim and metadata of self and other much be the same.
        """
        assert len(self.rep_dims) == 1
        assert len(other.rep_dims) == 1
        assert (
            self.rep_dims[0] == other.rep_dims[0]
        ), "The representation dimensions of self and other must be contiguous"
        odim = self.rep_dims[0]
        out_ten = self.ten.unsqueeze(odim + 1).mul(other.ten.unsqueeze(odim))
        out_metadata = torch.cat([self.metadata, other.metadata], dim=0)
        out_rep_layout = (
            self.rep_layout[0],
            other.rep_layout[0],
        )
        return self.__class__(
            out_ten,
            rep_dims=(odim, odim + 1),
            metadata=out_metadata,
            rep_layout=out_rep_layout,
            num_channels=(self.num_channels, other.num_channels),
        )

    def fold(self, stride: int, update_self=False) -> "SphericalTensor":
        """
        Fold the chucked representation channels of a 1-d SphericalTensor to a new dimension.
        """
        assert len(self.rep_dims) == 1
        assert torch.all(torch.fmod(self.metadata[0], stride) == 0), (
            f"The number of channels for the SphericalTensor to be folded must be multiples of "
            f"stride, got ({self.metadata}, {stride}) instead"
        )
        new_ten = self.ten.unflatten(
            dim=self.rep_dims[0],
            sizes=(self.shape[self.rep_dims[0]] // stride, stride),
        )
        new_metadata = self.metadata // stride
        new_rep_layout = (self.rep_layout[0][:, ::stride],)
        new_num_channels = (self.num_channels[0] // stride,)
        if update_self:
            self.ten = new_ten
            self.metadata = new_metadata
            self.rep_layout = new_rep_layout
            self.num_channels = new_num_channels
            return self
        else:
            return self.__class__(
                new_ten,
                rep_dims=self.rep_dims,
                metadata=new_metadata,
                rep_layout=new_rep_layout,
                num_channels=new_num_channels,
            )

    def unfold(self, update_self=False) -> "SphericalTensor":
        """
        Contract one trailing dimension into the representation dimension.
        """
        assert len(self.rep_dims) == 1
        assert self.ten.dim() > self.rep_dims[0], "No trailing dimension to unfold"
        stride = self.ten.shape[self.rep_dims[0] + 1]
        new_ten = self.ten.flatten(
            self.rep_dims[0],
            self.rep_dims[0] + 1,
        )
        new_metadata = self.metadata * stride
        new_rep_layout = (self.rep_layout[0].repeat_interleave(stride, dim=1),)
        new_num_channels = (self.num_channels[0] * stride,)
        if update_self:
            self.ten = new_ten
            self.metadata = new_metadata
            self.rep_layout = new_rep_layout
            self.num_channels = new_num_channels
            return self
        else:
            return self.__class__(
                new_ten,
                rep_dims=self.rep_dims,
                metadata=new_metadata,
                rep_layout=new_rep_layout,
                num_channels=new_num_channels,
            )

    def transpose_repdims(self, inplace=False):
        assert (
            len(self.rep_dims) == 2
        ), "transpose_repdims only supports 2d SphericalTensor"
        ten_t = torch.transpose(self.ten, *self.rep_dims).contiguous()
        dims_t = self.rep_dims
        metadata_t = self.metadata[(1, 0), :]
        rep_layout_t = self.rep_layout[::-1]
        num_channels_t = self.num_channels[::-1]
        if inplace:
            self.ten = ten_t
            self.rep_dims = dims_t
            self.metadata = metadata_t
            self.rep_layout = rep_layout_t
            self.num_channels = num_channels_t
            return self
        else:
            return self.__class__(
                ten_t,
                rep_dims=dims_t,
                metadata=metadata_t,
                rep_layout=rep_layout_t,
                num_channels=num_channels_t,
            )

    def invariant(self) -> torch.Tensor:
        """
        Returns the invariant content of a SphericalTensor

        When self.n_rep_dim==1, the l=0 channels are retained;
        when self.n_rep_dim==2, the (l1=0, l2=0) channels are also contracted.
        """
        if len(self.rep_dims) == 1:
            l0_length = self.metadata[0, 0]
            ops_dim = self.rep_dims[0]
            data_l0 = torch.narrow(self.ten, dim=ops_dim, start=0, length=l0_length)
            norm_shape = list(self.shape)
            norm_shape[ops_dim] = self.num_channels[0] - l0_length
            data_rep = torch.narrow(
                self.ten,
                dim=ops_dim,
                start=l0_length,
                length=self.ten.shape[ops_dim] - l0_length,
            )
            # Subtract the L=0 offset in the pointer tensor
            idx_ten = self.rep_layout[0][2, l0_length:].sub(l0_length)
            invariant_rep = NormContraction1d.apply(
                data_rep, idx_ten, norm_shape, ops_dim, self._norm_eps
            )
            return torch.cat([data_l0, invariant_rep], dim=ops_dim)
        elif len(self.rep_dims) == 2:
            idx_ten_0 = (
                self.rep_layout[0][2, :]
                .unsqueeze(1)
                .expand(
                    self.ten.shape[self.rep_dims[0]], self.ten.shape[self.rep_dims[1]]
                )
            )
            idx_ten_1 = (
                self.rep_layout[1][2, :]
                .unsqueeze(0)
                .expand(
                    self.ten.shape[self.rep_dims[0]], self.ten.shape[self.rep_dims[1]]
                )
            )
            idx_tens = torch.stack([idx_ten_0, idx_ten_1], dim=0)
            norm_shape = list(self.shape)
            norm_shape[self.rep_dims[0]] = self.num_channels[0]
            norm_shape[self.rep_dims[1]] = self.num_channels[1]
            invariant2d = NormContraction2d.apply(
                self.ten, idx_tens, norm_shape, self.rep_dims, self._norm_eps
            )
            return invariant2d
        else:
            raise NotImplementedError

    def generate_rep_layout(self) -> Tuple[torch.LongTensor, ...]:
        if len(self.rep_dims) == 1:
            return (self.generate_rep_layout_1d_(self.metadata[0]).to(self.ten.device),)
        elif len(self.rep_dims) == 2:
            rep_layout_0 = self.generate_rep_layout_1d_(self.metadata[0]).to(self.ten.device)
            rep_layout_1 = self.generate_rep_layout_1d_(self.metadata[1]).to(self.ten.device)
            return rep_layout_0, rep_layout_1
        else:
            raise NotImplementedError

    @staticmethod
    def generate_rep_layout_1d_(metadata1d) -> torch.LongTensor:
        n_irreps_per_l = torch.arange(start=0, end=metadata1d.size(0)) * 2 + 1
        end_channelids = torch.cumsum(metadata1d, dim=0)
        start_channelids = torch.cat([torch.LongTensor([0]), end_channelids[:-1]])
        dst_ls = torch.repeat_interleave(
            torch.arange(n_irreps_per_l.size(0)), n_irreps_per_l * metadata1d
        )
        dst_ms = torch.repeat_interleave(
            torch.cat(
                [
                    torch.arange(-l, l + 1)
                    for l in torch.arange(start=0, end=metadata1d.size(0))
                ]
            ),
            torch.repeat_interleave(metadata1d, n_irreps_per_l),
        )
        ns = torch.arange(metadata1d.sum())
        dst_ns = torch.cat(
            [
                ns[start_channelids[l] : end_channelids[l]].repeat(n_irreps)
                for l, n_irreps in enumerate(n_irreps_per_l)
            ]
        )
        rep_layout = torch.stack([dst_ls, dst_ms, dst_ns], dim=0).long()
        assert rep_layout.shape[1] == torch.sum(n_irreps_per_l * metadata1d)

        return rep_layout

    def to(self, device):
        # Do not transfer metadata to GPU, as index tracking should be more efficient
        # on CPUs
        self.ten = self.ten.to(device)
        self.rep_layout = tuple(layout.to(device) for layout in self.rep_layout)
        return self


@dataclass
class O3Tensor(SphericalTensor):
    """
    The O3Tensor class tracks the O(3) representation indices with additional
    parity (p) index tracker.

    Two indexing tensors are maintained, ``metadata`` and ``rep_layout``.

    Attributes:
        data_ten (torch.Tensor): The tensor storing auto-differentiable data.
        metadata (torch.LongTensor): The minimal information which specifies
            the number of non-degenerate feature channels for each dimension
            and each l and p (and the associated m(s)).
        rep_layout (Tuple[torch.LongTensor]): Stores the pointer (l[i], m[i], n[i], p[i])
            for each index i along the SO(3) representation dimension of the
            flattened data tensor.

    Args:
        data_ten (torch.Tensor): the underlying flattened data tensor in the
             strictly incremental [l,m,-p,n] order, for which the rep_dim has N elements
        rep_dims (tuple of ints): the spherical tensor representation dimension(s). When
             n_rep_dims > 1, must be contiguous dimensions.
        metadata (torch.LongTensor): (n_rep_dims x (2 * n_l)) specification of the number of unique
             channels (n) for each angular momentum index l.
        rep_layout (tuple of ints, optional): the (4xN) index tensor for (l[i],m[i],n[i],p[i]).
        num_channels (tuple of ints, optional): the number of unique channels per rep_dim.
    """

    def __init__(
        self,
        data_ten: torch.Tensor,
        rep_dims: Tuple[int, ...],
        metadata: torch.LongTensor,
        rep_layout: Optional[Tuple[torch.LongTensor, ...]] = None,
        num_channels: Optional[Tuple[int, ...]] = None,
    ):
        self.ten = data_ten
        self.metadata = metadata
        assert self.metadata.shape[1] % 2 == 0
        self.rep_dims = rep_dims
        assert self.rep_dims[-1] < self.ten.dim()
        self._norm_eps = 1e-4
        if rep_layout:
            self.rep_layout = rep_layout
        else:
            if len(rep_dims) > 1:
                assert rep_dims[1] - rep_dims[0] == 1
            self.rep_layout = self.generate_rep_layout()
        if num_channels:
            self.num_channels = num_channels
        else:
            self.num_channels = tuple(torch.sum(self.metadata, dim=1).long().tolist())

    def self_like(self, new_data_ten: torch.Tensor):
        """
        Args:
            new_data_ten: The underlying data of the new O3Tensor.

        Returns:
             A O3Tensor with identical layout but new data tensor.
        """
        return O3Tensor(
            new_data_ten,
            rep_dims=self.rep_dims,
            metadata=self.metadata,
            rep_layout=self.rep_layout,
            num_channels=self.num_channels,
        )

    @staticmethod
    def generate_rep_layout_1d_(metadata1d) -> torch.LongTensor:
        num_distinct_ls = metadata1d.size(0) // 2
        n_irreps_per_l = torch.arange(start=0, end=num_distinct_ls) * 2 + 1
        n_irreps_per_lp = n_irreps_per_l.repeat_interleave(2)
        ls_metadata = torch.arange(start=0, end=num_distinct_ls).repeat_interleave(2)
        end_channelids = torch.cumsum(metadata1d, dim=0)
        start_channelids = torch.cat([torch.LongTensor([0]), end_channelids[:-1]])
        dst_ls = torch.repeat_interleave(ls_metadata, n_irreps_per_lp * metadata1d)
        dst_ms = torch.repeat_interleave(
            torch.cat([torch.arange(-l, l + 1) for l in ls_metadata]),
            torch.repeat_interleave(metadata1d, n_irreps_per_lp),
        )
        ns = torch.arange(metadata1d.sum())
        dst_ns = torch.cat(
            [
                ns[start_channelids[l] : end_channelids[l]].repeat(n_irreps)
                for l, n_irreps in enumerate(n_irreps_per_lp)
            ]
        )
        ps = torch.LongTensor([1, -1]).repeat(num_distinct_ls)
        dst_ps = torch.repeat_interleave(ps, n_irreps_per_lp * metadata1d)
        rep_layout = torch.stack([dst_ls, dst_ms, dst_ns, dst_ps], dim=0).long()
        assert rep_layout.shape[1] == torch.sum(n_irreps_per_lp * metadata1d)

        return rep_layout

    @classmethod
    def from_so3(cls, so3_ten: SphericalTensor, parity=1):
        metadata = so3_ten.metadata
        if parity == 1:
            o3_metadata = torch.stack(
                [metadata, torch.zeros_like(metadata)], dim=2
            ).view(metadata.shape[0], -1)
            o3_layout = tuple(
                torch.cat([layout, torch.ones(1, layout.shape[1], device=layout.device)], dim=0)
                for layout in so3_ten.rep_layout
            )
        elif parity == -1:
            o3_metadata = torch.stack(
                [torch.zeros_like(metadata), metadata], dim=2
            ).view(metadata.shape[0], -1)
            o3_layout = tuple(
                torch.cat([layout, torch.ones(1, layout.shape[1], device=layout.device).neg()], dim=0)
                for layout in so3_ten.rep_layout
            )
        else:
            raise NotImplementedError

        return O3Tensor(
            so3_ten.ten,
            rep_dims=so3_ten.rep_dims,
            metadata=o3_metadata,
            rep_layout=o3_layout,
            num_channels=so3_ten.num_channels,
        )
