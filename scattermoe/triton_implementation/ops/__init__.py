import torch

from .compileable_ops import compileable_bincount, group, group_bwd_W, scatter2scatter


BLOCK_M = 128
torch._dynamo.config.capture_scalar_outputs = True


def expert_boundaries(sorted_experts_idxs: torch.Tensor, k: int):
    # there is an overhead of launching a custom op so we only use the custom op when compiling
    if torch.compiler.is_compiling():
        expert_counts = compileable_bincount(sorted_experts_idxs, k)
    else:
        expert_counts = sorted_experts_idxs.bincount(minlength=k)
    expert_boundaries_end = expert_counts.cumsum(-1)
    return expert_boundaries_end


class _ScatteredExperts(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        expert_weights,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        gates=None,
        grouped_in=False,
        grouped_out=False,
    ):
        output = torch.empty(sorted_expert_idxs.size(0), expert_weights.size(-1), device=x.device, dtype=x.dtype)

        scatter2scatter(
            X=x,
            W=expert_weights,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            out=output,
            FAN_OUT=k,
            x_grouped=grouped_in,
            y_grouped=grouped_out,
        )

        if gates is None:
            output_expanded = None
        else:
            output_expanded = output.view(gates.size(0), gates.size(1), output.size(-1))
            output = torch.bmm(gates.unsqueeze(1), output_expanded).squeeze(1)

        ctx.save_for_backward(
            x,
            expert_weights,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            gates,
            output_expanded,
        )

        ctx.grouped_in = grouped_in
        ctx.grouped_out = grouped_out
        ctx.k = k

        return output

    @staticmethod
    def backward(ctx, grad_out):
        (
            x,
            expert_weights,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            gates,
            output_expanded,
        ) = ctx.saved_tensors
        k = ctx.k
        grouped_in = ctx.grouped_in
        grouped_out = ctx.grouped_out

        if gates is None:
            d_gates = None
            gates_flat = None
            gate_fan = 1
            grouped_grad_out = None
        else:
            # calculate gates gradient
            d_gates = torch.bmm(output_expanded, grad_out.unsqueeze(2)).squeeze(-1)
            gates_flat = gates.flatten()
            gate_fan = gates.size(1)
            # print("expanded and grouping")
            grouped_grad_out = output_expanded.flatten(0, 1)  # reuse expanded buffer later

        if grouped_out:
            grouped_grad_out = grad_out
        else:
            group(
                A=grad_out,
                sorted_expert_idxs=sorted_scattered_idxs,
                out=grouped_grad_out,
                coeff=gates_flat,
                fan_out=gate_fan,
            )

        if grouped_in:
            grouped_x = x
            d_expanded_input = torch.empty(
                sorted_expert_idxs.size(0), expert_weights.size(1), device=x.device, dtype=x.dtype
            )
        else:
            grouped_x = torch.empty(sorted_scattered_idxs.size(0), x.size(1), dtype=x.dtype, device=x.device)
            group(
                A=x,
                sorted_expert_idxs=sorted_scattered_idxs,
                out=grouped_x,
                fan_out=k,
            )

            d_expanded_input = grouped_x

        d_weights = torch.zeros(
            expert_weights.size(0),
            grouped_grad_out.size(-1),
            grouped_x.size(-1),
            device=grouped_grad_out.device,
            dtype=grouped_grad_out.dtype,
        ).permute(0, 2, 1)

        group_bwd_W(
            DY=grouped_grad_out,
            X=grouped_x,
            expert_offsets=expert_offsets,
            DW=d_weights,
            E=expert_weights.size(0),
        )

        scatter2scatter(
            X=grouped_grad_out,
            W=expert_weights.permute(0, 2, 1),
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            out=d_expanded_input,
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
            d_input,
            d_weights,
            None,
            # sorted_expert_idxs, sorted_scattered_idxs,
            None,
            None,
            # expert_offsets,
            None,
            # gates
            d_gates,
            None,
            None,
        )


def scattered_experts(
    inputs,
    expert_weights,
    k,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    expert_offsets,
    gates=None,
    grouped_in=False,
    grouped_out=False,
):
    x = inputs
    expert_weights = expert_weights
    k = k
    sorted_expert_idxs = sorted_expert_idxs
    sorted_scattered_idxs = sorted_scattered_idxs
    expert_offsets = expert_offsets
    gates = gates
    grouped_in = grouped_in
    grouped_out = grouped_out

    return _ScatteredExperts.apply(
        x,
        expert_weights,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        gates,
        grouped_in,
        grouped_out,
    )
