# ScatterMoE Mixtral

Example integration of ScatterMoE into HuggingFace's implementation of Mixtral. 
We replace `MixtralSparseMoeBlock`([original source](https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/mixtral/modeling_mixtral.py#L816)) with a ScatterMoE implementation ([source](https://github.com/shawntan/scattermoe/blob/main/examples/mixtral/modeling_mixtral.py#L667)). 

We do not support loading of the existing Mixtral model for now, but to initialise a model from scratch:
```python
config = MixtralConfig.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True,
    attn_implementation='flash_attention_2'
)
```
for training:
```python
config.output_router_logits = True
```
This will ensure that the auxiliary loss is added to the loss computed for training. The MoE auxiliary losses are
load balancing losses that try to that there is no over-reliance on only a few experts during training.



