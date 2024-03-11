# scattermoe
Triton-based implementation of Sparse Mixture of Experts. 

## Installation
```sh
# Check all is working well.
PYTHONPATH=. pytests tests
# Install editable. This will allow you to modify scattermoe in this directory.
pip install -e .
```
Enjoy!

## Usage
```python
from scattermoe.mlp import MoE

# Initialise module...
mlp = MoE(
    input_size=x_dim, hidden_size=h_dim,
    activation=nn.GELU(),
    num_experts=E, top_k=k
)

# Calling module...
Y = mlp(
    X,         # input tensor
    k_weights, # top-k weights from router
    k_idxs     # top-k indices from router
)
