import torch

from torch_gauge.o3.spherical import SphericalTensor


class IEILinear(torch.nn.Module):
    r"""
    Irrep-wise Equivariant Linear Layer

    This module takes a spherical tensor and perform linear transformation within
     the feature channels spanned by each irreducible representation index, (l, m):

    \f[
        \mathbf{h}^{\textrm{out}}_{l,m} \mathbf{W}^{l} \cdot \mathbf{h}^{\textrm{in}}_{l,m}
    \f]

    The matrix multiplication is always performed on the last dimension of the spherical tensors.

    To vectorize: the coefficient matrix \f$\mathbf{W}^{l}\f$ is stored as a flattened 1D tensor,
     together with a block-sparse layout tensor in the COO format; the coefficients
     are gathered to the block-sparse matrix and converted to the CSR format in runtime.
     May be only beneficial when the batch size is small.

    In the future, we should make it a CUDA kernel to avoid serial operations or redundant
     type conversions for sparse matrices. The first optimization we should do is to
     enable ILP by executing multiple CUDA streams.
    Attributes:
        metadata_in (torch.LongTensor): the number of non-degenerate feature channels of the input tensor
         for each l (and the associated m(s)).
        metadata_out (torch.LongTensor): the number of non-degenerate feature channels of the output tensor
         for each l (and the associated m(s)).a
    Returns:
        The data tensor of the output spherical tensor.
    """

    def __init__(self, metadata_in, metadata_out):

        super().__init__()
        assert len(metadata_in) == len(metadata_out)
        self._metadata_in = metadata_in
        self._metadata_out = metadata_out
        # The bias must be False
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(metadata_in[l], metadata_out[l], bias=False) for l, _ in enumerate(metadata_in)]
        )
        n_irreps_per_l = torch.arange(start=0, end=metadata_in.size(0)) * 2 + 1
        self._end_inds_in = torch.cumsum(self._metadata_in * n_irreps_per_l, dim=0)
        self._start_inds_in = (torch.cat([torch.LongTensor([0]), self._end_inds_in[:-1]]),)

    def forward(self, x: SphericalTensor) -> torch.Tensor:

        assert x.metadata == self._metadata_in
        # TODO: vectorization
        outs = []
        for l, linear_l in enumerate(self._metadata_in):
            in_l = x.ten[..., self._start_inds_in[l] : self._end_inds_in[l]]
            out_l = linear_l(in_l.view(2 * l + 1, -1)).view(-1)
            outs.append(out_l)

        return torch.cat(outs, dim=-1)
