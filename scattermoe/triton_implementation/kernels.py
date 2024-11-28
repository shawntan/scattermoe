import triton
import triton.language as tl
import torch

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

    # iters = E_last_idx - E_first_idx + 1
    # for i in range(iters):
    #     E_idx = i + E_first_idx
    for E_idx in range(E_first_idx, E_last_idx + 1):
        E_mask = E_idxs == E_idx
        # E_M_idx = tl.where(E_mask, M_idx, 0)
        E_M_idx = M_idx
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
def groupXtY_triton_kernel_back(
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
    # pid0, pid1 = tl.swizzle2d(pid0, pid1, num0, num1, 4)
    pid1, pid0 = tl.swizzle2d(pid1, pid0, num1, num0, 128)

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



def heuristic_init(meta):
    result = torch.zeros(
        (meta['DW_ptr'].size(0),
         triton.cdiv(meta['K'], meta['BLOCK_K']),
         triton.cdiv(meta['N'], meta['BLOCK_N'])),
        dtype=torch.int32, device=meta['DW_ptr'].device
    )
    return result



@triton.autotune(
    # configs=[
        # triton.Config({"BLOCK_N": bn, "BLOCK_K": bk, "BLOCK_M": bmi * bm_factor, "BLOCK_M_INNER": bmi},
                    #   num_stages=num_stages, num_warps=num_warps)
        # for bn in [64, 128]
        # for bk in [64, 128]
        # for bmi in [64, 128]
        # for bm_factor in [4, 8, 16]
        # for num_stages in [2, 4, 6]
        # for num_warps in [4, 8]
    # ],
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64, "BLOCK_M": 1024, "BLOCK_M_INNER": 64},
                     num_stages=6, num_warps=4)
    ],
    key=["N", "K", "E"],
)
@triton.heuristics(
    values={
        "DW_count_ptr": heuristic_init,
        "DW_lock_ptr": heuristic_init,
        "NO_K_MASK": lambda meta: meta['K'] % meta['BLOCK_K'] == 0,
        "NO_N_MASK": lambda meta: meta['N'] % meta['BLOCK_N'] == 0
    }
)

@triton.jit
def groupXtY_triton_kernel(
    DY_ptr, stride_dym, stride_dyk,
    X_ptr, stride_xm, stride_xn,
    DW_ptr, stride_dwe, stride_dwk, stride_dwn,
    expert_offsets_ptr,
    sorted_expert_idxs_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_M_INNER: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NO_K_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    allow_tf32: tl.constexpr,
    DW_count_ptr, DW_lock_ptr, 
    SWIZZLE_FACTOR: tl.constexpr=128,
):

    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    pid2 = tl.program_id(axis=2)
    num0 = tl.num_programs(0)
    num1 = tl.num_programs(1)
    # num2 = tl.num_programs(2)
    # pid0, pid1 = tl.swizzle2d(pid0, pid1, num0, num1, 4)
    pid1, pid0 = tl.swizzle2d(pid1, pid0, num1, num0, SWIZZLE_FACTOR)
    # K_BLOCK_COUNT = tl.cdiv(K, BLOCK_K)
    K_block_id = pid0
    N_block_id = pid1
    M_block_id = pid2


    K_block = K_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    K_mask = K_block < K
    # K_block = tl.max_contiguous(tl.multiple_of(K_block % K, BLOCK_K), BLOCK_K)

    N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_block < N
    # N_block = tl.max_contiguous(tl.multiple_of(N_block % N, BLOCK_N), BLOCK_N)
    M_block = M_block_id * BLOCK_M + tl.arange(0, BLOCK_M_INNER)

    E_M_first_idx = tl.load(sorted_expert_idxs_ptr + M_block_id * BLOCK_M).to(tl.int32)
    if ((M_block_id + 1) * BLOCK_M - 1) < M:
        E_M_last_idx = tl.load(sorted_expert_idxs_ptr + (M_block_id + 1) * BLOCK_M - 1).to(tl.int32)
    else:
        E_M_last_idx = E - 1


    xt_blk_ptrs = X_ptr + K_block[:, None] * stride_xn + M_block[None, :] * stride_xm
    dy_blk_ptrs = DY_ptr + M_block[:, None] * stride_dym + N_block[None, :] * stride_dyk

    iters = BLOCK_M // BLOCK_M_INNER
    prev_E_idx = -1
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=ACC_TYPE)
    DW_block_ptrs = DW_ptr + K_block[:, None] * stride_dwk + N_block[None, :] * stride_dwn
    KN_mask = K_mask[:, None] & N_mask[None, :]

    M_inner_first_idx = M_block_id * BLOCK_M
    M_inner_last_idx = (M_block_id + 1) * BLOCK_M - 1
    if E_M_first_idx == E_M_last_idx: # If entire BLOCK_M is one expert.
        E_idx = E_M_first_idx
        prev_E_idx = E_idx
        for i in tl.range(iters):
            M_mask = M_block < M
            no_m_mask = M_inner_last_idx < M

            dy, xt = load_dy_xt(xt_blk_ptrs, dy_blk_ptrs, M_mask, K_mask, N_mask,no_m_mask, NO_K_MASK, NO_N_MASK)
            acc = tl.dot(xt, dy, acc=acc, out_dtype=ACC_TYPE, allow_tf32=allow_tf32)

            M_inner_last_idx += BLOCK_M_INNER
            M_block += BLOCK_M_INNER
            xt_blk_ptrs += BLOCK_M_INNER * stride_xm
            dy_blk_ptrs += BLOCK_M_INNER * stride_dym

    else:
        skip_check = False

        for i in tl.range(iters):
            M_mask = M_block < M
            no_m_mask = M_inner_first_idx + BLOCK_M_INNER - 1 < M
            dy, xt = load_dy_xt(xt_blk_ptrs, dy_blk_ptrs, M_mask, K_mask, N_mask,no_m_mask, NO_K_MASK, NO_N_MASK)

            E_first_idx = tl.load(sorted_expert_idxs_ptr + M_inner_first_idx).to(tl.int32)
            if M_inner_first_idx + BLOCK_M_INNER - 1 < M:
                E_last_idx = tl.load(sorted_expert_idxs_ptr + M_inner_first_idx + BLOCK_M_INNER - 1).to(tl.int32)
            else:
                E_last_idx = E - 1

            """
            if skip_check:
                E_first_idx = E_M_last_idx
                E_last_idx = E_M_last_idx
            else:
                E_first_idx = tl.load(sorted_expert_idxs_ptr + M_inner_first_idx).to(tl.int32)

                if E_first_idx == E_M_last_idx:
                    E_last_idx = E_first_idx
                    skip_check = True
                else:
                    if M_inner_first_idx + BLOCK_M_INNER - 1 < M:
                        E_last_idx = tl.load(sorted_expert_idxs_ptr + M_inner_first_idx + BLOCK_M_INNER - 1).to(tl.int32)
                    else:
                        E_last_idx = E - 1
                if E_last_idx == E_M_last_idx:
                    skip_check = True
            """


            if E_first_idx == E_last_idx: # If entire BLOCK_M_INNER is one expert
                E_idx = E_last_idx
                if prev_E_idx != -1 and E_idx != prev_E_idx: # if E_idx changed, write to HBM
                    locked_add(
                        Lock_ptr=DW_lock_ptr + K_block_id * num1 + N_block_id + prev_E_idx * num0 * num1,
                        Count_ptr=DW_count_ptr + K_block_id * num1 + N_block_id + prev_E_idx * num0 * num1,
                        A_ptrs=DW_block_ptrs + prev_E_idx * stride_dwe,
                        a=acc,
                        mask=KN_mask,
                        NO_MASK=NO_K_MASK and NO_N_MASK,
                        NO_LOCK=E_M_first_idx < prev_E_idx and prev_E_idx < E_M_last_idx
                    )
                    acc = tl.zeros_like(acc)
                acc = tl.dot(xt, dy, acc=acc, out_dtype=ACC_TYPE, allow_tf32=allow_tf32)
                prev_E_idx = E_idx
            else:
                if no_m_mask:
                    E_idxs = tl.load(sorted_expert_idxs_ptr + M_block).to(tl.int32)
                else:
                    E_idxs = tl.load(sorted_expert_idxs_ptr + M_block, mask=M_mask, other=E).to(tl.int32)

                for E_idx in range(E_first_idx, E_last_idx + 1):
                    E_mask = E_idxs == E_idx
                    # dy_ = tl.where(E_mask[:, None], dy, 0.)
                    xt_ = tl.where(E_mask[None, :], xt, 0.).to(xt.dtype)
                    if prev_E_idx != -1 and E_idx != prev_E_idx: # if E_idx changed, write to HBM
                        locked_add(
                            Lock_ptr=DW_lock_ptr + prev_E_idx * num0 * num1 + K_block_id * num1 + N_block_id,
                            Count_ptr=DW_count_ptr + prev_E_idx * num0 * num1 + K_block_id * num1 + N_block_id,
                            A_ptrs=DW_block_ptrs + prev_E_idx * stride_dwe,
                            a=acc,
                            mask=KN_mask,
                            NO_MASK=NO_K_MASK and NO_N_MASK,
                            NO_LOCK=E_M_first_idx < prev_E_idx and prev_E_idx < E_M_last_idx
                        )
                        acc = tl.zeros_like(acc)
                    acc = tl.dot(xt_, dy, acc=acc, out_dtype=ACC_TYPE, allow_tf32=allow_tf32)
                    prev_E_idx = E_idx

            M_inner_first_idx += BLOCK_M_INNER
            M_inner_last_idx += BLOCK_M_INNER
            M_block += BLOCK_M_INNER
            xt_blk_ptrs += BLOCK_M_INNER * stride_xm
            dy_blk_ptrs += BLOCK_M_INNER * stride_dym

    locked_add(
        Lock_ptr=DW_lock_ptr + prev_E_idx * num0 * num1 + K_block_id * num1 + N_block_id,
        Count_ptr=DW_count_ptr + prev_E_idx * num0 * num1 + K_block_id * num1 + N_block_id,
        A_ptrs=DW_block_ptrs + prev_E_idx * stride_dwe,
        a=acc,
        mask=KN_mask,
        NO_MASK=NO_K_MASK and NO_N_MASK
    )

@triton.jit
def load_dy_xt(xt_blk_ptrs, dy_blk_ptrs, M_mask, K_mask, N_mask, no_m_mask, no_k_mask: tl.constexpr, no_n_mask: tl.constexpr):
    if no_m_mask:
        if no_n_mask:
            dy = tl.load(dy_blk_ptrs)
        else:
            dy = tl.load(dy_blk_ptrs, mask=N_mask[None, :])
        if no_k_mask:
            xt = tl.load(xt_blk_ptrs)
        else:
            xt = tl.load(xt_blk_ptrs, mask=K_mask[:, None])
    else:
        if no_n_mask:
            dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None])
        else:
            dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None] & N_mask[None, :])
        if no_k_mask:
            xt = tl.load(xt_blk_ptrs, mask=M_mask[None, :])
        else:
            xt = tl.load(xt_blk_ptrs, mask=M_mask[None, :] & K_mask[:, None])
    return dy, xt



@triton.jit
def locked_add(Lock_ptr, Count_ptr, A_ptrs, a, mask, NO_MASK, NO_LOCK=False):
    if NO_LOCK:
        if NO_MASK:
            tl.store(A_ptrs, a)
        else:
            tl.store(A_ptrs, a, mask=mask)
    else:
        locked = tl.atomic_cas(Lock_ptr, 0, 1)
        while locked == 1:
            locked = tl.atomic_cas(Lock_ptr, 0, 1)

        # BEGIN SINGLE THREAD
        count = tl.load(Count_ptr)
        if NO_MASK:
            if count == 0:
                tl.store(A_ptrs, a)
                tl.store(Count_ptr, 1)
            else:
                tl.store(A_ptrs, a + tl.load(A_ptrs))
        else:
            if count == 0:
                tl.store(A_ptrs, a, mask=mask)
                tl.store(Count_ptr, 1)
            else:
                tl.store(A_ptrs, a + tl.load(A_ptrs, mask=mask), mask=mask)
        # END SINGLE THREAD
        tl.atomic_xchg(Lock_ptr, 0)


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
