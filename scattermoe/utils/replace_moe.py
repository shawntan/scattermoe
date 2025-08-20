import torch
from .. import kernels, parallel_experts
from torch.nn import functional as F

def replace_function(cls, fun_name):
    def decorator(fun):
        import logging
        def _fun(*args, **kwargs):
            filename = fun.__code__.co_filename
            name = fun.__name__
            logging.info(f"Replacing `{cls.__name__}.{fun_name}` with {filename}:{name}")
            setattr(cls, fun_name, fun)
            return fun(*args, **kwargs)
        setattr(cls, fun_name, _fun)

    return decorator
            

try:
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts, GptOssMLP, GptOssForCausalLM
    @replace_function(cls=GptOssExperts, fun_name='forward')
    def gpt_oss_forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        k = router_indices.shape[1]
        selected_weights = torch.gather(routing_weights, dim=1, index=router_indices)
        with torch.no_grad():
            router_indices_ = router_indices.flatten()
            sorted_expert_idxs, sorted_scattered_idxs = kernels.ops.flatten_and_sort(router_indices_)
            padded_block_idxs, expert_offsets = kernels.ops.padded_block_indices(sorted_expert_idxs, self.num_experts)
        gate_up = parallel_experts.parallel_linear(
            hidden_states, self.gate_up_proj, k,
            sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            grouped_in=False, grouped_out=True, 
        )
        gate_up = gate_up + self.gate_up_proj_bias[sorted_expert_idxs]

        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated_output_ = (up + 1) * glu

        out_scattered = parallel_experts.parallel_linear(
            gated_output_, self.down_proj, 1,
            sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            grouped_in=True, grouped_out=False,
        )
        out_scattered = out_scattered + self.down_proj_bias[router_indices_]
        out_scattered = out_scattered.view(out_scattered.size(0) // k, k, -1)
        out_ = (selected_weights.unsqueeze(1) @ out_scattered).squeeze(1)
        next_states = out_.view(batch_size, -1, self.hidden_size)

        return next_states
except:
    pass


try: 
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import GraniteMoeHybridMoE
    @replace_function(cls=GraniteMoeHybridMoE, fun_name='forward')
    def granite_moe_forward(self, layer_input):
        bsz, length, emb_size = layer_input.size()
        layer_input = layer_input.reshape(-1, emb_size)
        # compute the top_k routing decision
        router_logits = self.router.layer(layer_input)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.router.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(layer_input.dtype)
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = kernels.ops.flatten_and_sort(selected_experts)
            padded_block_idxs, expert_offsets = kernels.ops.padded_block_indices(sorted_expert_idxs, self.router.num_experts)

        # compute experts
        gates, h = parallel_experts.parallel_linear(
            layer_input, self.input_linear.weight.transpose(2, 1),
            self.router.top_k,
            sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            grouped_in=False, grouped_out=True, 
        ).chunk(2, dim=-1)
        h = self.activation(gates) * h
        layer_output = parallel_experts.parallel_linear(
            h, self.output_linear.weight.transpose(2, 1),
            1,
            sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            grouped_in=True, grouped_out=False,
            gates=routing_weights
        )
        layer_output = layer_output.view(bsz, length, emb_size)
        return layer_output, router_logits
except:
    pass


# def swap_moe_modules(model: GptOssForCausalLM):
#     for mod in model.modules():
#         if isinstance(mod, GptOssMLP):
#             new_experts = object.__new__(GptOssScatteredExperts)
#             new_experts.clone_params(mod.experts)
#             mod.experts = new_experts
#             print("replaced GptOssExpert")


