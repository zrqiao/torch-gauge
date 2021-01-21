"""
Native implementation of point-cloud based graph neural networks
 with torch-gauge functionalities
"""

import math

import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter

from torch_gauge.verlet_list import VerletList
from torch_gauge.nn import SSP


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
            Linear(_nf, _nf), SSP(), Linear(_nf, _nf), SSP()
        )
        self.pre_conv = Linear(self._nf, self._nf)
        self.post_conv = torch.nn.Sequential(Linear(_nf, _nf), SSP(), Linear(_nf, _nf))

    def forward(self, vl: VerletList, l: int):
        pre_conv = self.pre_conv(vl.ndata[f"atomic_{l}"])
        d_ij = (vl.query_src(vl.ndata["xyz"]) - vl.ndata["xyz"].unsqueeze(1)).norm(
            dim=2, keepdim=True
        )
        conv_out = self.cfconv(
            torch.exp(-self.gamma * (d_ij - self.rbf_centers.view(1, 1, -1)).pow(2))
        )
        vl.ndata[f"atomic_{l+1}"] = vl.ndata[f"atomic_{l}"] + self.post_conv(
            conv_out * pre_conv
        )
        return vl.ndata[f"atomic_{l+1}"]
