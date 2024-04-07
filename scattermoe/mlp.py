import torch
from torch import nn
from torch.nn import functional as F

from . import kernels
from .parallel_experts import ParallelExperts

class GLUMLP(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size,
        num_experts,
        top_k,
        activation=nn.SiLU(),
    ):
        super(GLUMLP, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.experts = ParallelExperts(num_experts, input_size, 2 * hidden_size)
        self.output_experts = ParallelExperts(num_experts, hidden_size, input_size)
        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

    def extra_repr(self):
        return 'k={}'.format(self.top_k)

    def forward(self, x: torch.Tensor, expert_p: torch.Tensor, expert_idxs: torch.Tensor):
        x_shape = x.size()
        x = x.view(-1, x_shape[-1])
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = kernels.ops.flatten_and_sort(expert_idxs)
            padded_block_idxs, expert_offsets = kernels.ops.padded_block_indices(sorted_expert_idxs, self.num_experts)

        def check_vals(x):
            assert not (torch.isnan(x) | torch.isinf(x)).any()
    
        h, gates  = self.experts(
            x, self.top_k,
            sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            grouped_out=True
        ).chunk(2, dim=-1)
        h = self.activation(gates) * h
        check_vals(h)
        y = self.output_experts(
            h, 1, sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            grouped_in=True,
            gates=expert_p,
        )
        return y


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_experts,
        top_k,
        activation=None,
    ):
        super(MLP, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.experts = ParallelExperts(num_experts, input_size, hidden_size)
        self.output_experts = ParallelExperts(num_experts, hidden_size, input_size)
        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

    def extra_repr(self):
        return 'k={}'.format(self.top_k)

    def forward(self, x: torch.Tensor, expert_p: torch.Tensor, expert_idxs: torch.Tensor):
        x_shape = x.size()
        x = x.view(-1, x_shape[-1])
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = kernels.ops.flatten_and_sort(expert_idxs)
            padded_block_idxs, expert_offsets = kernels.ops.padded_block_indices(sorted_expert_idxs, self.num_experts)

        h = self.experts(
            x, self.top_k,
            sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            grouped_out=True
        )
        h = self.activation(h)
        y = self.output_experts(
            h, 1, sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            grouped_in=True,
            gates=expert_p,
        )
        y = y.view(*x_shape[:-1], y.size(-1))
        return y
