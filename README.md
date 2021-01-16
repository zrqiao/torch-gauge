# Torch-Gauge

A light-weight [PyTorch](https://pytorch.org/) extension for efficient gauge-equivariant learning.

Torch-Gauge is a library to boost geometric learning on physical data structures
with Lie-group symmetry and beyond two-body interactions. GPU and mini-batch training
utilities are natively supported.

## Setups

### From Source
Once PyTorch is installed, running 

    pip install -e .

## Project Structure
- `torch_gauge` High-level python functions
    - `torch_gauge/o3` O(3) group algebra functionals
    - `torch_gauge/nn.py` Tensorial neural network building blocks
    - `torch_gauge/verlet_list.py` Verlet neighbor-list for representing relational data
    - `torch_gauge/tests` contains pytests
- `src` contains C++/CUDA backend operations, under development
    
## Test

    make test

## Questions?

Please submit a question/issue or [email me](mailto:zqiao@caltech.edu).
