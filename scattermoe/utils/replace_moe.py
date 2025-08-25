import torch
from .. import parallel_linear, flatten_sort_count
from torch.nn import functional as F
from torch import nn

import logging

def replace_function(cls, fun_name):
    def decorator(fun):
        def _fun(*args, **kwargs):
            filename = fun.__code__.co_filename
            name = fun.__name__
            logging.info(f"Replacing `{cls.__name__}.{fun_name}` with {filename}:{name}")
            setattr(cls, fun_name, fun)
            return fun(*args, **kwargs)
        setattr(cls, fun_name, _fun)
    return decorator

try:
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
    @replace_function(cls=GptOssExperts, fun_name='forward')
    def gpt_oss_forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        k = router_indices.shape[1]
        selected_weights = torch.gather(routing_weights, dim=1, index=router_indices)
        router_indices = router_indices.flatten()
        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = \
            flatten_sort_count(router_indices, num_experts=self.num_experts)

        gate_up = parallel_linear(
            hidden_states, self.gate_up_proj, k,
            sorted_expert_idxs, sorted_scattered_idxs,
            expert_offsets,
            expert_biases=self.gate_up_proj_bias,
            grouped_in=False, grouped_out=True, 
        )

        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated_output_ = (up + 1) * glu

        out_scattered = parallel_linear(
            gated_output_, self.down_proj, 1,
            sorted_expert_idxs, sorted_scattered_idxs,
            expert_offsets,
            expert_biases=self.down_proj_bias,
            grouped_in=True, grouped_out=False,
            gates=selected_weights,
        )

        next_states = out_scattered.view(batch_size, -1, self.hidden_size)
        return next_states
except Exception:
    logging.info("Failed to replace GptOssExperts")


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
        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = \
            flatten_sort_count(selected_experts, num_experts=self.router.num_experts)

        # compute experts
        gates, h = parallel_linear(
            layer_input, self.input_linear.weight.transpose(2, 1),
            self.router.top_k,
            sorted_expert_idxs, sorted_scattered_idxs,
            expert_offsets,
            grouped_in=False, grouped_out=True,
        ).chunk(2, dim=-1)
        h = self.activation(gates) * h
        layer_output = parallel_linear(
            h, self.output_linear.weight.transpose(2, 1),
            1,
            sorted_expert_idxs, sorted_scattered_idxs,
            expert_offsets,
            grouped_in=True, grouped_out=False,
            gates=routing_weights
        )
        layer_output = layer_output.view(bsz, length, emb_size)
        return layer_output, router_logits
except Exception:
    logging.info("Failed to replace GraniteMoeHybridMoE")

# TODO consolidating params into tensor. OOMs.
# try: 
#     from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
#     fun = Qwen3MoeSparseMoeBlock.__init__
#     def qwen3moe__init__(self: Qwen3MoeSparseMoeBlock, config):
#         fun(self, config)

#         weight_refs = {}
#         def assemble_weights(self):
#             down_proj_weights = [weight_refs[i, "down_proj"] for i in range(len(self.experts))]
#             up_proj_weights = [weight_refs[i, "up_proj"] for i in range(len(self.experts))]
#             gate_proj_weights = [weight_refs[i, "gate_proj"] for i in range(len(self.experts))]
#             if (all(w is not None for w in down_proj_weights) and
#                 all(w is not None for w in up_proj_weights) and
#                 all(w is not None for w in gate_proj_weights)):
#                 self.act_fn = self.experts[0].act_fn
#                 mid_size = up_proj_weights[0].size(0)
#                 self.down_proj = nn.Parameter(torch.stack(down_proj_weights))
#                 gate_up_proj = torch.empty(
#                     self.num_experts, 2 * mid_size, config.hidden_size,
#                     dtype=up_proj_weights[0].dtype
#                 )
#                 for i in range(self.num_experts - 1, -1, -1):
#                     gate_up_proj[i, :mid_size] = gate_proj_weights[i]
#                     gate_up_proj[i, mid_size:] = up_proj_weights[i]
#                     del self.experts[i]
#                 self.gate_up_proj = nn.Parameter(gate_up_proj)
#                 del self.experts
        
#         for i, e in enumerate(self.experts):
#             def weight_tracker(expert_id, expert_weight_name):
#                 def fun(module, err_msgs):
#                     id_tup = (expert_id, expert_weight_name)
#                     assert id_tup in weight_refs
#                     weight_refs[id_tup] = module.weight
#                     assemble_weights(self)
#                 return fun
#             e.gate_proj.register_load_state_dict_post_hook(weight_tracker(i, "gate_proj"))
#             weight_refs[i, "gate_proj"] = None
#             e.up_proj.register_load_state_dict_post_hook(weight_tracker(i, "up_proj"))
#             weight_refs[i, "up_proj"] = None
#             e.down_proj.register_load_state_dict_post_hook(weight_tracker(i, "down_proj"))
#             weight_refs[i, "down_proj"] = None

#         # def hook(module, state_dict, prefix, local_metadata):
#         #     local_keys = [k for k in state_dict.keys() if k.startswith(prefix)]
#         #     expert_keys = [k for k in local_keys if 'mlp.' in k]
#         #     if len(expert_keys) == 0: # deleted
#         #         assert prefix + "down_proj" in state_dict
#         #         assert prefix + "gate_up_proj" in state_dict
#         # self.register_state_dict_post_hook(hook)
    
#     state_dict_fun = Qwen3MoeSparseMoeBlock.state_dict
#     def qwen3moe_state_dict(self: Qwen3MoeSparseMoeBlock, *args, destination=None, prefix="", keep_vars=False):
#         destination = state_dict_fun(self, *args, destination, prefix, keep_vars)
#         local_keys = [k for k in destination.keys() if k.startswith(prefix)]
#         if ((prefix + "down_proj" in local_keys) and (prefix + "gate_up_proj" in local_keys)):
#             # need to break down
#             for i in range(self.num_experts):
#                 down_proj_name = prefix + f"experts.{i}.down_proj.weight"
#                 up_proj_name = prefix + f"experts.{i}.up_proj.weight"
#                 gate_proj_name = prefix + f"experts.{i}.gate_proj.weight"
#                 destination[down_proj_name] = destination[prefix + "down_proj"][i]
#                 gate_proj, up_proj = destination[prefix + "gate_up_proj"][i].chunk(2, dim=1)
#                 destination[gate_proj_name] = gate_proj.contiguous()
#                 destination[up_proj_name] = up_proj.contiguous()
#             del destination[prefix + "down_proj"]
#             del destination[prefix + "gate_up_proj"]

#         return destination
#     Qwen3MoeSparseMoeBlock.state_dict = qwen3moe_state_dict


#     Qwen3MoeSparseMoeBlock.__init__ = qwen3moe__init__

#     def qwen3moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         """ """
#         batch_size, sequence_length, hidden_dim = hidden_states.shape
#         hidden_states = hidden_states.view(-1, hidden_dim)
#         # router_logits: (batch * sequence_length, n_experts)
#         router_logits = self.gate(hidden_states)

#         routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
#         routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
#         if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
#             routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
#         # we cast back to the input dtype
#         routing_weights = routing_weights.to(hidden_states.dtype)
#         sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = flatten_sort_count(selected_experts, num_experts=self.num_experts)

#         gate_up = parallel_linear(
#             hidden_states, self.gate_up_proj.transpose(1, 2), self.top_k,
#             sorted_expert_idxs, sorted_scattered_idxs,
#             expert_offsets,
#             grouped_in=False, grouped_out=True, 
#         )

#         _gate, up = gate_up.chunk(2, dim=-1)
#         intermediate = self.act_fn(_gate) * up

#         final_hidden_states = parallel_linear(
#             intermediate, self.down_proj.transpose(1, 2), 1,
#             sorted_expert_idxs, sorted_scattered_idxs,
#             expert_offsets,
#             grouped_in=True, grouped_out=False,
#             gates=routing_weights,
#         )
#         final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
#         return final_hidden_states, router_logits
#     Qwen3MoeSparseMoeBlock.forward = qwen3moe_forward
# except Exception as e:
#     pass

