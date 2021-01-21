import torch

from torch_gauge.o3.spherical import SphericalTensor


class SSP(torch.nn.Softplus):
    """Shifted SoftPlus activation"""

    def __init__(self, beta=1, threshold=20):
        super().__init__(beta, threshold)

    def forward(self, input):
        return F.softplus(input, self.beta, self.threshold) - math.sqrt(2)


class Swish_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(torch.nn.Module):
    def forward(self, input_tensor):
        return Swish_fn.apply(input_tensor)


class IELin(torch.nn.Module):
    r"""
    Irrep-wise Equivariant Linear Layer

    This module takes a spherical tensor and perform linear transformation within
     the feature channels spanned by each irreducible representation index, (l, m):

    \f[
        \mathbf{h}^{\textrm{out}}_{l,m} = \mathbf{W}^{l} \cdot \mathbf{h}^{\textrm{in}}_{l,m}
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
        assert metadata_in.dim() == 1
        assert len(metadata_in) == len(metadata_out)
        self._metadata_in = metadata_in
        self._metadata_out = metadata_out
        n_irreps_per_l = torch.arange(start=0, end=metadata_in.size(0)) * 2 + 1
        # The bias must be False
        self.linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    metadata_in[l] * n_irreps_per_l[l],
                    metadata_out[l] * n_irreps_per_l[l],
                    bias=False,
                )
                if self._metadata_out[l] > 0
                else None
                for l, _ in enumerate(metadata_in)
            ]
        )
        self._end_inds_in = torch.cumsum(self._metadata_in * n_irreps_per_l, dim=0)
        self._start_inds_in = torch.cat([torch.LongTensor([0]), self._end_inds_in[:-1]])
        self.out_layout = SphericalTensor.generate_rep_layout_1d_(self._metadata_out)
        self.num_out_channels = torch.sum(self._metadata_out).item()

    def forward(self, x: SphericalTensor) -> SphericalTensor:
        assert x.rep_dims[-1] == x.ten.dim() - 1
        assert torch.all(x.metadata[-1].eq(self._metadata_in)), (
            f"Expected the SphericalTensor x and self._metadata_in to have the "
            f"same irrep metadata along the last dimension, got {x.metadata[-1]}"
            f" and {self._metadata_in} instead"
        )
        # TODO: vectorization
        outs = []
        for l, linear_l in enumerate(self.linears):
            if linear_l is None:
                continue
            in_l = x.ten[
                ..., self._start_inds_in[l] : self._end_inds_in[l]
            ].contiguous()
            out_l = linear_l(in_l)
            outs.append(out_l)

        out_ten = torch.cat(outs, dim=-1)
        out_metadata = x.metadata.clone()
        out_metadata[-1] = self._metadata_out
        out_rep_layout = x.rep_layout[:-1] + (self.out_layout,)
        return SphericalTensor(
            out_ten,
            rep_dims=x.rep_dims,
            metadata=out_metadata,
            rep_layout=out_rep_layout,
            num_channels=x.num_channels[:-1] + (self.num_out_channels,),
        )


class RepNorm1d(torch.nn.Module):
    r"""
    The (experimental) Representation Normalization layer.

    \f[
        RepNorm(\mathbf{h})_{l,m} = \frac{BatchNorm(||\mathbf{h}_l||)}{\sqrt{2l+1}} \cdot
         \frac{\mathbf{h}_{l}}{ |(1-\bm{\beta})||\mathbf{h}_{l}|| + \bm{\beta}| + \epsilon}
    \f]

    Heuristically, the trainable \f$\beta\f$ controls the fraction of norm information to be retained.

    Returns:
        normed_ten: the normed invariant content
        gauge_ten: the "pure gauge" spherical tensor
    """

    def __init__(self, num_channels, momentum=0.1, eps=1e-2):
        super().__init__()
        self._num_channels = num_channels
        self._eps = eps
        self.batchnorm = torch.nn.BatchNorm1d(
            num_features=num_channels, momentum=momentum
        )
        # TODO: initialization schemes
        self.beta = torch.nn.Parameter(torch.rand(self._num_channels))

    def forward(self, x: SphericalTensor) -> (torch.Tensor, SphericalTensor):
        x0 = x.invariant()
        assert x0.dim() == 2
        x1 = self.batchnorm(x0)
        divisor = torch.abs(x0.mul(1 - self.beta) + self.beta) + self._eps
        x2 = x.scalar_mul(1 / divisor)
        return x1, x2
