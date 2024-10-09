import triton
import triton.language as tl


BLOCK_M = 128


@triton.autotune(
    configs=[triton.Config({"BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4)],
    key=["N", "K"],
)
@triton.jit
def scatter2scatter_triton_kernel(
    X_ptr, stride_xm, stride_xk,
    W_ptr, stride_we, stride_wk, stride_wn,
    Y_ptr, stride_ym, stride_yn,
    grouped_idx_ptr,
    expert_idxs_ptr,
    # block_start_idx_ptr,
    FAN_OUT,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    allow_tf32: tl.constexpr,
    x_grouped,
    y_grouped,
):
    pid = tl.program_id(axis=0)

    N_BLOCK_COUNT = tl.cdiv(N, BLOCK_N)
    M_block_id = pid // N_BLOCK_COUNT
    N_block_id = pid % N_BLOCK_COUNT
    M_range = tl.arange(0, BLOCK_M)
    N_range = tl.arange(0, BLOCK_N)
    # block_start_idx = tl.load(block_start_idx_ptr + M_block_id)

    # M_block = tl.max_contiguous(M_block_id * BLOCK_M + M_range, BLOCK_M)
    M_block = M_block_id * BLOCK_M + M_range
    N_block = N_block_id * BLOCK_N + N_range
    N_mask = N_block < N
    M_boundary_mask = M_block < (FAN_OUT * M)
    E_idxs = tl.load(expert_idxs_ptr + M_block, mask=M_boundary_mask, other=E)

    no_k_mask = K % BLOCK_K == 0
    no_n_mask = N % BLOCK_N == 0

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    E_first_idx = tl.min(E_idxs)
    E_last_idx = tl.minimum(tl.max(E_idxs), E - 1)
    M_idx = tl.load(grouped_idx_ptr + M_block, mask=M_boundary_mask, other=0).to(tl.int32)

    iters = E_last_idx - E_first_idx + 1
    for i in range(iters):
        E_idx = i + E_first_idx
        E_mask = E_idxs == E_idx
        E_M_idx = tl.where(E_mask, M_idx, 0)
        if x_grouped:
            M_in_idx = M_block
        else:
            M_in_idx = E_M_idx // FAN_OUT

        acc = compute_expert_block(
            E_idx, E_mask,
            M_in_idx,
            N_block, N_mask,
            X_ptr, stride_xm, stride_xk,
            W_ptr, stride_we, stride_wk, stride_wn,
            K,
            acc,
            allow_tf32,
            no_k_mask, no_n_mask,
            ACC_TYPE, BLOCK_K
        )
    if y_grouped:
        M_out_idx = M_block
    else:
        M_out_idx = M_idx

    Y_blk_ptrs = Y_ptr + (M_out_idx[:, None] * stride_ym + N_block[None, :] * stride_yn)
    tl.store(Y_blk_ptrs, acc, mask=M_boundary_mask[:, None] & N_mask[None, :])

@triton.jit
def compute_expert_block(
        E_idx, E_mask,
        M_in_idx,
        N_block, N_mask,
        X_ptr, stride_xm, stride_xk,
        W_ptr, stride_we, stride_wk, stride_wn,
        K,
        acc,
        allow_tf32,
        no_k_mask, no_n_mask,
        ACC_TYPE, BLOCK_K):

    K_block = tl.arange(0, BLOCK_K)
    X_blk_ptrs = X_ptr + M_in_idx[:, None] * stride_xm + K_block[None, :] * stride_xk
    W_blk_ptrs = W_ptr + K_block[:, None] * stride_wk + N_block[None, :] * stride_wn + E_idx * stride_we
    iters = tl.cdiv(K, BLOCK_K)

    for K_block_id in range(0, iters):
        if no_k_mask:
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None])
            if no_n_mask or K_block_id < (iters - 1):
                w = tl.load(W_blk_ptrs)
            else:
                w = tl.load(W_blk_ptrs, mask=N_mask[None, :])
        else:
            K_mask = (K_block_id * BLOCK_K + K_block) < K
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None] & K_mask[None, :])
            w = tl.load(W_blk_ptrs, mask=K_mask[:, None] & N_mask[None, :])

        X_blk_ptrs += BLOCK_K * stride_xk
        W_blk_ptrs += BLOCK_K * stride_wk
        acc += tl.dot(x, w, allow_tf32=allow_tf32, out_dtype=ACC_TYPE)

    return acc


@triton.autotune(
    configs=[
        # different block M and reducing stages
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_M": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_M": 128}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_M": 64}, num_stages=2, num_warps=4),
        # keep 4 stages and keep two 64 block sizes
        # - NOTE: these can get good performances for low M, but for large M the variation
        # triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64, 'BLOCK_M': 64}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'BLOCK_M': 64}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'BLOCK_M': 64}, num_stages=4, num_warps=4),
    ],
    key=["N", "K"],
)
@triton.jit
def groupXtY_triton_kernel(
    DY_ptr,
    stride_dym,
    stride_dyk,
    X_ptr,
    stride_xm,
    stride_xn,
    DW_ptr,
    stride_dwe,
    stride_dwk,
    stride_dwn,
    expert_offsets_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    num0 = tl.num_programs(0)
    num1 = tl.num_programs(1)
    pid0, pid1 = tl.swizzle2d(pid0, pid1, num0, num1, 4)

    K_BLOCK_COUNT = tl.cdiv(K, BLOCK_K)
    E_idx = pid0 // K_BLOCK_COUNT
    K_block_id = pid0 % K_BLOCK_COUNT
    N_block_id = pid1

    if E_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(expert_offsets_ptr + E_idx - 1).to(tl.int32)

    end_idx = tl.load(expert_offsets_ptr + E_idx).to(tl.int32)

    if end_idx > start_idx:
        M_block = tl.max_contiguous(start_idx + tl.arange(0, BLOCK_M), BLOCK_M)

        K_block = K_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
        K_mask = K_block < K
        K_block = tl.max_contiguous(tl.multiple_of(K_block % K, BLOCK_K), BLOCK_K)

        N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        N_mask = N_block < N
        N_block = tl.max_contiguous(tl.multiple_of(N_block % N, BLOCK_N), BLOCK_N)

        M_idxs = M_block
        xt_blk_ptrs = X_ptr + K_block[:, None] * stride_xn + M_idxs[None, :] * stride_xm
        dy_blk_ptrs = DY_ptr + M_idxs[:, None] * stride_dym + N_block[None, :] * stride_dyk

        acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=ACC_TYPE)

        iters = tl.cdiv(end_idx - start_idx, BLOCK_M)

        no_k_mask = K % BLOCK_K == 0
        no_n_mask = N % BLOCK_N == 0

        for i in range(0, iters):
            M_mask = (i * BLOCK_M + M_block) < end_idx

            if no_k_mask:
                xt = tl.load(xt_blk_ptrs, mask=M_mask[None, :])
            else:
                xt = tl.load(xt_blk_ptrs, mask=K_mask[:, None] & M_mask[None, :])

            if no_n_mask:
                dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None])
            else:
                dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None] & N_mask[None, :])

            xt_blk_ptrs += BLOCK_M * stride_xm
            dy_blk_ptrs += BLOCK_M * stride_dym
            acc += tl.dot(xt, dy, out_dtype=ACC_TYPE, allow_tf32=allow_tf32)

        DW_blk_ptrs = DW_ptr + E_idx * stride_dwe + K_block[:, None] * stride_dwk + N_block[None, :] * stride_dwn
        acc = acc.to(DW_blk_ptrs.dtype.element_ty)
        tl.store(DW_blk_ptrs, acc, mask=K_mask[:, None] & N_mask[None, :])


@triton.autotune(configs=[triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=4, num_warps=4)], key=["K"])
@triton.jit
def group_triton_kernel(
    src_ptr,
    stride_sn,
    stride_sk,
    has_coeff: tl.constexpr,
    coeff_ptr,
    FAN_OUT,
    tgt_ptr,
    stride_tn,
    stride_ti,
    grouped_idx_ptr,
    N,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    N_block_id = pid
    N_blk = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_blk < N
    N_blk = tl.max_contiguous(tl.multiple_of(N_blk % N, BLOCK_N), BLOCK_N)
    N_idx = tl.load(grouped_idx_ptr + N_blk, mask=N_mask, other=0)

    K_blk = tl.arange(0, BLOCK_K)
    src_blk_ptrs = src_ptr + (N_idx // FAN_OUT)[:, None] * stride_sn + K_blk[None, :] * stride_sk
    tgt_blk_ptrs = tgt_ptr + N_blk[:, None] * stride_tn + K_blk[None, :] * stride_ti

    if has_coeff:
        c = tl.load(coeff_ptr + N_idx, mask=N_mask)[:, None]

    iters = tl.cdiv(K, BLOCK_K)
    no_k_mask = K % BLOCK_K == 0

    for i in range(0, iters):
        if no_k_mask or i < iters - 1:
            block = tl.load(src_blk_ptrs, mask=N_mask[:, None])

            if has_coeff:
                block *= c

            tl.store(tgt_blk_ptrs, block, mask=N_mask[:, None])
        else:
            K_mask = (i * BLOCK_K + K_blk) < K
            mask = N_mask[:, None] & K_mask[None, :]
            block = tl.load(src_blk_ptrs, mask=mask)

            if has_coeff:
                block *= c

            tl.store(tgt_blk_ptrs, block, mask=mask)

        src_blk_ptrs += BLOCK_K * stride_sk
        tgt_blk_ptrs += BLOCK_K * stride_ti
