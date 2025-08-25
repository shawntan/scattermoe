# scattermoe
Triton-based implementation of Sparse Mixture-of-Experts (SMoE) on GPUs.
ScatterMoE builds upon existing implementations, and overcoming some of the limitations to improve inference, training speed, and memory footprint. 
This implementation achieves this by avoiding padding and making excessive copies of the input.
We also fuse expert linear transforms and reordering operations with `ParallelLinear`, a module that can be used to extend the concept of SMoEs.

This implementation is lightweight (~700 lines).
It will work within an FSDP or pipeline parallel framework, but does not include any additional multi-node training infrastructure code.
You can find the report [here](https://arxiv.org/abs/2403.08245)

## Installation
```sh
# Check all is working well.
PYTHONPATH=. pytest tests
# Install editable. This will allow you to modify scattermoe in this directory.
pip install -e .
```

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
```

## Bibtex
If you use ScatterMoE in your project, cite us!
```bibtex
@article{tan2024scattered,
  title={Scattered Mixture-of-Experts Implementation},
  author={Tan, Shawn and Shen, Yikang and Panda, Rameswar and Courville, Aaron},
  journal={arXiv preprint arXiv:2403.08245},
  year={2024}
}
```

Enjoy!
----

###  Version 0.3.0

- Refactored away padded indices
- Allow bias
- Inject MoE implementation into the following implementations by `import scattermoe.utils.replace_moe`
  - `transformers.models.gpt_oss.modeling_gpt_oss` 
  - `transformers.models.granitemoehybrid.modeling_granitemoehybrid`

###  Version 0.2.0

- Made compileable.
