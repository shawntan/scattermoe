from typing import Callable

import torch
import torch.nn as nn

from .ops import padded_block_indices, scattered_experts


class ParallelExperts(nn.Module):
    def __init__(self, num_experts, input_size, output_size) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, output_size, input_size))
        self.reset_parameters()
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

    def extra_repr(self):
        return 'num_experts={}, input_size={}, output_size={}'.format(
            self.num_experts, self.input_size, self.output_size)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=0.02)

    def forward(
        self,
        inputs,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        padded_block_idxs,
        expert_offsets,
        gates=None,
        grouped_in=False,
        grouped_out=False,
    ):
        return scattered_experts(
            inputs,
            self.weight.permute(0, 2, 1),
            k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates,
            grouped_in,
            grouped_out,
        )

