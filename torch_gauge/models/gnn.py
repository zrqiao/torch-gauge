"""
Native implementation of point-cloud based graph neural networks
 with torch-gauge functionalities
"""


import torch
from torch.nn import Linear, Parameter

from torch_gauge.nn import SSP
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
            Linear(300, _nf), SSP(), Linear(_nf, _nf), SSP()
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
        vl.ndata[f"atomic_{l+1}"] = vl.ndata[f"atomic_{l}"] + self.post_conv(conv_out)
        return vl.ndata[f"atomic_{l+1}"]
