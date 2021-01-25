"""
Native implementation of point-cloud based graph neural networks
 with torch-gauge functionalities
"""


import torch
from torch.nn import Linear, Parameter

from torch_gauge.geometric import poly_env
from torch_gauge.nn import SSP, IELin
from torch_gauge.o3.clebsch_gordan import LeviCivitaCoupler
from torch_gauge.o3.rsh import RSHxyz
from torch_gauge.o3.spherical import SphericalTensor
from torch_gauge.verlet_list import VerletList


class SchNetLayer(torch.nn.Module):
    """
    Schütt, Kristof, et al. "Schnet: A continuous-filter convolutional neural network for modeling quantum interactions."
     Advances in neural information processing systems. 2017.
    """

    def __init__(self, num_features):
        super().__init__()
        _nf = num_features
        self.gamma = 10.0
        self.rbf_centers = Parameter(
            torch.linspace(0.1, 30.1, 300), requires_grad=False
        )
        self.cfconv = torch.nn.Sequential(
            Linear(300, _nf, bias=False), SSP(), Linear(_nf, _nf, bias=False), SSP()
        )
        self.pre_conv = Linear(_nf, _nf)
        self.post_conv = torch.nn.Sequential(Linear(_nf, _nf), SSP(), Linear(_nf, _nf))

    def forward(self, vl: VerletList, l: int):
        xyz = vl.ndata["xyz"]
        pre_conv = self.pre_conv(vl.ndata[f"atomic_{l}"])
        d_ij = (vl.query_src(xyz) - xyz.unsqueeze(1)).norm(dim=2, keepdim=True)
        filters = self.cfconv(
            torch.exp(-self.gamma * (d_ij - self.rbf_centers.view(1, 1, -1)).pow(2))
        )
        conv_out = (filters * vl.query_src(pre_conv) * vl.edge_mask.unsqueeze(2)).sum(1)
        vl.ndata[f"atomic_{l + 1}"] = vl.ndata[f"atomic_{l}"] + self.post_conv(conv_out)
        return vl.ndata[f"atomic_{l + 1}"]


class SE3Layer(torch.nn.Module):
    """
    A generic rotational-equivariant layer restricted to l<=1, similar to:
    Thomas, Nathaniel, et al. "Tensor field networks: Rotation-and translation-equivariant neural networks for
     3d point clouds." arXiv preprint arXiv:1802.08219 (2018).
    """

    def __init__(self, num_features):
        super().__init__()
        _nf = num_features
        self.rbf_freqs = Parameter(torch.arange(16), requires_grad=False)
        self.rsh_mod = RSHxyz(max_l=1)
        self.coupler = LeviCivitaCoupler(torch.LongTensor([_nf, _nf]))
        self.filter_gen = torch.nn.Sequential(
            Linear(16, _nf, bias=False), SSP(), Linear(_nf, _nf, bias=False)
        )
        self.pre_conv = IELin([_nf, _nf], [_nf, _nf])
        self.post_conv = torch.nn.ModuleList(
            [
                IELin([2 * _nf, 3 * _nf], [_nf, _nf]),
                SSP(),
                IELin([_nf, _nf], [_nf, _nf]),
            ]
        )

    def forward(self, vl: VerletList, l: int):
        r_ij = vl.query_src(vl.ndata["xyz"]) - vl.ndata["xyz"].unsqueeze(1)
        d_ij = r_ij.norm(dim=2, keepdim=True)
        r_ij = r_ij / d_ij
        feat_in: SphericalTensor = vl.ndata[f"atomic_{l}"]
        pre_conv = vl.query_src(self.pre_conv(feat_in))
        filters_radial = self.filter_gen(
            torch.sin(d_ij / 5 * self.rbf_freqs.view(1, 1, -1))
            / d_ij
            * poly_env(d_ij / 5)
        )
        filters = pre_conv.self_like(
            self.rsh_mod(r_ij).ten.unsqueeze(-1).mul(filters_radial).flatten(2, 3)
        )
        coupling_out = self.post_conv[0](
            self.coupler(pre_conv, filters, overlap_out=False)
        )
        conv_out = feat_in.self_like(
            coupling_out.ten.mul(vl.edge_mask.unsqueeze(2)).sum(1)
        )
        conv_out = conv_out.scalar_mul(self.post_conv[1](conv_out.invariant()))
        conv_out.ten = self.post_conv[2](conv_out).ten
        vl.ndata[f"atomic_{l+1}"] = feat_in + conv_out
        return vl.ndata[f"atomic_{l+1}"]
