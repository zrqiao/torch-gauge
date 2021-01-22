# Torch-Gauge

A light-weight [PyTorch](https://pytorch.org/) extension for efficient gauge-equivariant learning.

## About
**Torch-Gauge** is a library to boost geometric learning on physical data structures
with Lie-group symmetry and beyond two-body interactions. The library is designed to be specially optimized
for training and inference on GPUs, with mini-batch training utilities natively supported.

## Usage
Torch-Gauge uses padded [Verlet list](https://en.wikipedia.org/wiki/Verlet_list) as the core for 
manipulating relational data, which enables compact and highly customizable implementation of 
geometric learning models.

As an illustration, check out our implementation of [SchNet](https://arxiv.org/abs/1706.08566) 's
 interaction module in just 20 lines of code:
```python
import torch
from torch.nn import Linear, Parameter
from torch_gauge.verlet_list import VerletList
from torch_gauge.nn import SSP

class SchNetLayer(torch.nn.Module):
  def __init__(self, num_features):
    super().__init__()
    _nf = num_features
    self.gamma = 10.0
    self.rbf_centers = Parameter(torch.linspace(0.1, 30.1, 300), requires_grad=False)
    self.cfconv = torch.nn.Sequential(Linear(300, _nf), SSP(), Linear(_nf, _nf), SSP())
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
```
And that's it.

## Setups

### From Source
Once PyTorch is installed, running 

    pip install versioneer && versioneer install
    pip install -e .

## Project Structure
- `torch_gauge` High-level python functions
    - `torch_gauge/o3` O(3) group algebra functionals
    - `torch_gauge/nn.py` Tensorial neural network building blocks
    - `torch_gauge/verlet_list.py` Verlet neighbor-list operations for representing relational data
    - `torch_gauge/geometric.py` contains geometric and algebra operations with autograd
    - `torch_gauge/models` contains exemplary implementations of GNN variants and descriptors 
    - `torch_gauge/tests` contains pytests
- `src` contains C++/CUDA backend operations (under development)
    
## Test

    make test

## Questions?

Please submit a question/issue or [email me](mailto:zqiao@caltech.edu).
