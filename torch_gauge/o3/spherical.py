from dataclasses import dataclass
from typing import Optional, Tuple
from torch_gauge.o3.functional import NormContraction1d

import torch


@dataclass
class SphericalTensor:
    r"""
    The SphericalTensor class tracks the SO(3) representation indices
     for a flattened data tensor for efficient group algebra operations.

    All angular and magnetic quantum number indices are treated as equivalent,
     and all non-degenerate feature channels are regarded as the "principal"
     indices, n. Note that the angular quantum number indices (m) for each l
     range from 0 to 2l, instead of -l to l in the "physicist's" convention.

    Two indexing tensors are maintained, ``metadata`` and ``rep_layout``.
    Attributes:
        data_ten (torch.Tensor): the tensor storing auto-differentiable data.
        metadata (torch.LongTensor): the minimal information which specifies
            the number of non-degenerate feature channels for each dimension
            and each l (and the associated m(s)).
        rep_layout (torch.LongTensor): stores the pointer (l[i], m[i], n[i])
            for each index i along the SO(3) representation dimension of the
            flattened data tensor.
    """

    def __init__(
        self,
        data_ten: torch.Tensor,
        rep_dims: Tuple[int],
        metadata: torch.LongTensor,
        rep_layout: Optional[torch.LongTensor] = None,
        num_channels: Optional[torch.LongTensor] = None,
    ):
        """
        Args:
            data_ten: the underlying flattened data tensor in the
             strictly incremental [l,m,n] order, for which the rep_dim has N elements
            rep_dims: the spherical tensor representation dimension(s). When
             n_rep_dims > 1, must be contiguous dimensions.
            metadata: (n_rep_dims x n_l) specification of the number of unique
             channels (n) for each angular momentum index l.
            rep_layout (optional): the (3xN) index tensor for (l[i],m[i],n[i]).
            num_channels (optional): the number of unique channels per rep_dim.
        """
        self.ten = data_ten
        self.metadata = metadata
        self.rep_dims = rep_dims
        self._norm_eps = 1e-4
        if rep_layout:
            self._rep_layout = rep_layout
        else:
            if len(rep_dims) > 1:
                assert rep_dims[1] - rep_dims[0] == 1
            self._rep_layout = self.generate_rep_layout()
        if num_channels:
            self.num_channels = num_channels
        else:
            self.num_channels = torch.sum(self.metadata, dim=1).long()

    @property
    def shape(self):
        return self.ten.shape

    def mul_(self, other: "SphericalTensor"):
        # Be careful that this operation will break equivariance
        self.ten.mul_(other.ten)
        return self.ten

    def add_(self, other: "SphericalTensor"):
        self.ten.add_(other.ten)
        return self.ten

    def __mul__(self, other: "SphericalTensor"):
        return SphericalTensor(
            self.ten * other.ten,
            rep_dims=self.rep_dims,
            metadata=self.metadata,
            rep_layout=self._rep_layout,
            num_channels=self.num_channels,
        )

    def __add__(self, other: "SphericalTensor"):
        return SphericalTensor(
            self.ten + other.ten,
            rep_dims=self.rep_dims,
            metadata=self.metadata,
            rep_layout=self._rep_layout,
            num_channels=self.num_channels,
        )

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
            broadcasted_other = torch.index_select(other, dim=self.rep_dims[0], index=self._rep_layout[0, 2, :])
        elif len(self.rep_dims) == 2:
            broadcasted_other = torch.index_select(
                other.view(
                    *other.shape[: self.rep_dims[0]],
                    self.num_channels[0] * self.num_channels[1],
                    *other.shape[self.rep_dims[1] + 1 :],
                ),
                dim=self.rep_dims[0],
                index=(
                    self._rep_layout[0, 2, :].unsqueeze(1) * self.num_channels[1]
                    + self._rep_layout[1, 2, :].unsqueeze(0)
                ).view(-1),
            ).view_as(self.ten)
        else:
            raise NotImplementedError

        if inplace:
            self.ten.mul_(broadcasted_other)
            return self
        else:
            return SphericalTensor(
                self.ten * broadcasted_other,
                rep_dims=self.rep_dims,
                metadata=self.metadata,
                rep_layout=self._rep_layout,
                num_channels=self.num_channels,
            )

    def dot(self, other: "SphericalTensor", dim: int):
        raise NotImplementedError

    def rep_dot(self, other: "SphericalTensor", dim: int):
        """
        Performing channel-wise inner multiplication.
        If self.n_rep_dim==1, a torch.Tensor is returned;
        if self.n_rep_dim==2, a SphericalTensor with n_rep_dim==1 is returned.
        Args:
            other (SphericalTensor): must be a 1-d spherical tensor
            with the same number of dimensions and broadcastable to self.ten.
            dim (int): the dimension to perform the inner product.
        """
        dotdim_idx = self.rep_dims.index(dim)
        assert other.rep_dims[0] == dim
        assert self.metadata[dotdim_idx] == other.metadata[0]
        singleton_shape = (-1 if d == dim else 1 for d in range(self.ten.dim()))
        mul_ten = self.ten * other.ten
        out_shape = mul_ten.shape
        out_shape[dim] = self.num_channels[dotdim_idx]
        out_ten = torch.zeros(out_shape, device=mul_ten.device, dtype=mul_ten.dtype).scatter_add_(
            dim=dim,
            index=self._rep_layout[dotdim_idx, 2, :].view(*singleton_shape).expand_as(mul_ten),
            src=mul_ten,
        )
        if len(self.rep_dims) == 1:
            return out_ten
        elif len(self.rep_dims) == 2:
            # A little trick here
            dimid_kept = 1 - dotdim_idx
            return SphericalTensor(
                out_ten,
                rep_dims=(self.rep_dims[dimid_kept],),
                metadata=self.metadata[dimid_kept].unsqueeze(0),
                rep_layout=self._rep_layout[dimid_kept].unsqueeze(0),
                num_channels=self.num_channels[dimid_kept].unsqueeze(0),
            )
        else:
            raise NotImplementedError

    def fold(self, stride: int, update_self=False) -> "SphericalTensor":
        """Fold the chucked representation channels to a new dimension"""
        assert len(self.rep_dims) == 1
        assert torch.all(torch.fmod(self.metadata, stride) == 0)
        new_ten = self.ten.unflatten(
            dim=self.rep_dims[0],
            sizes=(self.shape[self.rep_dims[0]]//stride, stride),
        )
        new_metadata = self.metadata//stride
        new_rep_layout = self._rep_layout[:, :, ::stride]
        new_num_channels = self.num_channels//stride
        if update_self:
            self.ten = new_ten
            self.metadata = new_metadata
            self._rep_layout = new_rep_layout
            self.num_channels = new_num_channels
            return self
        else:
            return SphericalTensor(
                new_ten,
                rep_dims=self.rep_dims,
                metadata=new_metadata,
                rep_layout=new_rep_layout,
                num_channels=new_num_channels,
            )

    def outer(self, other: "SphericalTensor") -> "SphericalTensor":
        """Returns the channel-wise outer product"""
        raise NotImplementedError

    def invariant(self) -> torch.Tensor:
        """
        Returns the invariant content
        When self.n_rep_dim==1, the l=0 channels are retained;
        When self.n_rep_dim==2, the (l1=0, l2=0) channels are also contracted.
        """
        if len(self.rep_dims) == 1:
            ops_dim = self.rep_dims[0]
            data_l0 = torch.narrow(self.ten, dim=ops_dim, start=0, length=self.metadata[0, 0])
            norm_shape = self.shape
            norm_shape[ops_dim] = self.num_channels[0]
            singleton_shape = (-1 if d == ops_dim else 1 for d in range(self.ten.dim()))
            data_rep = torch.narrow(
                self.ten,
                dim=ops_dim,
                start=self.metadata[0, 0],
                length=self.ten.shape[ops_dim] - self.metadata[0, 0],
            )
            idx_ten = self._rep_layout[0, 2, self.metadata[0, 0]:].view(singleton_shape).expand_as(data_rep.shape)
            invariant_rep = NormContraction1d.apply(data_rep, idx_ten, norm_shape, ops_dim, self._norm_eps)
            return torch.cat([data_l0, invariant_rep], dim=ops_dim)
        elif len(self.rep_dims) == 2:
            raise NotImplementedError

    def generate_rep_layout(self) -> torch.LongTensor:
        if len(self.rep_dims) == 1:
            return self._generate_rep_layout_1d(self.metadata[0]).unsqueeze(0).long()
        elif len(self.rep_dims) == 2:
            rep_layout_0 = self._generate_rep_layout_1d(self.metadata[0])
            rep_layout_1 = self._generate_rep_layout_1d(self.metadata[1])
            return torch.stack([rep_layout_0, rep_layout_1], dim=0).long()
        else:
            raise NotImplementedError

    @staticmethod
    def _generate_rep_layout_1d(metadata1d):
        n_irreps_per_l = torch.arange(start=0, end=metadata1d.size(0)) * 2 + 1
        end_channelids = torch.cumsum(metadata1d, dim=0)
        start_channelids = (torch.cat([torch.LongTensor([0]), end_channelids[:-1]]),)
        dst_ls = torch.repeat_interleave(torch.arange(n_irreps_per_l.size(0)), n_irreps_per_l * metadata1d)
        dst_ms = torch.repeat_interleave(
            torch.cat([torch.arange(0, n_irreps) for n_irreps in n_irreps_per_l]),
            torch.repeat_interleave(metadata1d, n_irreps_per_l),
        )
        dst_ns = torch.cat(
            [
                torch.arange(start_channelids[l], end_channelids[l]).repeat(n_irreps)
                for l, n_irreps in enumerate(n_irreps_per_l)
            ]
        )
        rep_layout = torch.stack([dst_ls, dst_ms, dst_ns], dim=0)
        assert rep_layout.shape[0] == torch.sum(n_irreps_per_l * metadata1d)

        return rep_layout

    def to(self, device):
        # Do not transfer metadata to GPU, as index tracking should be more efficient
        # on CPUs
        self.ten = self.ten.to(device)
        self._rep_layout = self._rep_layout.to(device)
        return self