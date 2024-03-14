# scattermoe
Triton-based implementation of Sparse Mixture of Experts. 

You can find the report [here](https://arxiv.org/abs/2403.08245)

## Installation
```sh
# Check all is working well.
PYTHONPATH=. pytest tests
# Install editable. This will allow you to modify scattermoe in this directory.
pip install -e .
```
Enjoy!

## Usage
```python
from scattermoe.mlp import MLP

# Initialise module...
mlp = MLP(
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
