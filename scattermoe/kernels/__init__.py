from . import compileable_ops as ops
from . import single

BLOCK_M = ops.BLOCK_M

def padded_block_indices(sorted_experts_idxs: torch.Tensor, k: int, N_BLOCK_SIZE: int = BLOCK_M):
    # there is an overhead of launching a custom op so we only use the custom op when compiling
    if torch.compiler.is_compiling():
        expert_counts = compileable_bincount(sorted_experts_idxs, k)
    else:
        expert_counts = sorted_experts_idxs.bincount(minlength=k)

    padded_block_counts = ((expert_counts - 1) // N_BLOCK_SIZE) + 1
    padded_expert_block_end = padded_block_counts.cumsum(-1)
    expert_boundaries_end = expert_counts.cumsum(-1)
    expert_boundaries_start = expert_boundaries_end - expert_counts
    padded_expert_block_start = padded_expert_block_end - padded_block_counts

    block_idxs = torch.arange(
        padded_expert_block_end[-1], dtype=sorted_experts_idxs.dtype, device=sorted_experts_idxs.device
    ).unsqueeze(1)

    block_mask = (block_idxs < padded_expert_block_start) | (block_idxs >= padded_expert_block_end)
    expanded_block_idxs = N_BLOCK_SIZE * (block_idxs - padded_expert_block_start) + expert_boundaries_start
    expanded_block_idxs = expanded_block_idxs.masked_fill(block_mask, 0).sum(-1)

    return expanded_block_idxs, expert_boundaries_end
