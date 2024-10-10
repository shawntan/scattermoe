import torch
import torch.nn as nn
from . import kernels

class ParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, expert_weights, k,
        sorted_expert_idxs, sorted_scattered_idxs,
        padded_block_idxs, expert_offsets,
        gates=None, grouped_in=False, grouped_out=False,
    ):
        with torch.device(x.device):
            output = torch.empty((sorted_expert_idxs.size(0), expert_weights.size(-1)),
                                 device=x.device, dtype=x.dtype)
            kernels.ops.scatter2scatter(
                X=x, W=expert_weights,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                padded_block_idxs=padded_block_idxs,
                out=output,
                FAN_OUT=k, x_grouped=grouped_in, y_grouped=grouped_out
            )
            if gates is None:
                output_expanded = None
            else:
                output_expanded = output.view(gates.size(0), gates.size(1), output.size(-1))
                output = torch.bmm(gates.unsqueeze(1), output_expanded).squeeze(1)

        ctx.save_for_backward(
            x, expert_weights,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            gates,
            output_expanded
        )
        ctx.grouped_in = grouped_in
        ctx.grouped_out = grouped_out
        ctx.k = k
        return output
    @staticmethod
    def backward(ctx, grad_out):
        (x, expert_weights,
         sorted_expert_idxs,
         sorted_scattered_idxs,
         padded_block_idxs, expert_offsets,
         gates, output_expanded) = ctx.saved_tensors
        k = ctx.k
        grouped_in = ctx.grouped_in
        grouped_out = ctx.grouped_out
        # print("backward")
        with torch.device(x.device):
            if gates is None:
                d_gates = None
                gates_flat = None
                gate_fan = 1
            else:
                # calculate gates gradient
                d_gates = torch.bmm(output_expanded, grad_out.unsqueeze(2)).squeeze(-1)
                gates_flat = gates.flatten()
                gate_fan = gates.size(1)

            if grouped_out:
                grouped_grad_out = grad_out
            else:
                grouped_grad_out = output_expanded.flatten(0, 1) # reuse expanded buffer later
                kernels.ops.group(
                    A=grad_out,
                    sorted_expert_idxs=sorted_expert_idxs,
                    out=grouped_grad_out,
                    coeff=gates_flat,
                    fan_out=gate_fan,
                )
            if grouped_in:
                grouped_x = x
                d_expanded_input = torch.empty(
                    (sorted_expert_idxs.size(0), expert_weights.size(1)),
                    device=x.device, dtype=x.dtype)
            else:
                grouped_x = torch.empty(
                    (sorted_scattered_idxs.size(0), x.size(1)),
                    dtype=x.dtype, device=x.device
                )
                kernels.ops.group(
                    A=x,
                    sorted_expert_idxs=sorted_scattered_idxs,
                    out=grouped_x,
                    fan_out=k
                )
                d_expanded_input = grouped_x

            d_weights = torch.zeros(
                expert_weights.size(0),
                grouped_grad_out.size(-1),
                grouped_x.size(-1),
                device=grouped_grad_out.device,
                dtype=grouped_grad_out.dtype,
            ).permute(0, 2, 1)

            kernels.ops.group_bwd_W(
                DY=grouped_grad_out,
                X=grouped_x,
                expert_offsets=expert_offsets,
                DW=d_weights,
                E=expert_weights.size(0)
            )

            kernels.ops.scatter2scatter(
                X=grouped_grad_out,
                W=expert_weights.permute(0, 2, 1),
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                padded_block_idxs=padded_block_idxs,
                out=d_expanded_input,  # Reuse grouped_x buffer
                FAN_OUT=1,
                x_grouped=True,
                y_grouped=grouped_in,
            )

            if k == 1:
                d_input = d_expanded_input
            else:
                d_input = d_expanded_input.view(x.size(0), k, d_expanded_input.size(-1)).sum(-2)
        # print("backward end.")
        return (
            # x, expert_weights, k,
            d_input, d_weights, None,
            # sorted_expert_idxs, sorted_scattered_idxs,
            None, None,
            # padded_block_idxs, expert_offsets,
            None, None,
            # gates
            d_gates, None, None
        )

def parallel_linear(inputs, expert_weights, k,
                    sorted_expert_idxs, sorted_scattered_idxs,
                    padded_block_idxs, expert_offsets,
                    gates=None, grouped_in=False, grouped_out=False):
    results = ParallelLinear.apply(inputs, expert_weights, k,
                                   sorted_expert_idxs, sorted_scattered_idxs,
                                   padded_block_idxs, expert_offsets, gates,
                                   grouped_in, grouped_out)
    return results

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

    def forward(self, inputs, k, sorted_expert_idxs, sorted_scattered_idxs,
                padded_block_idxs, expert_offsets,
                gates=None, grouped_in=False, grouped_out=False):

        results = parallel_linear(
            inputs, self.weight.permute(0, 2, 1), k,
            sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            gates=gates, grouped_in=grouped_in, grouped_out=grouped_out
        )
        return results
