from megablocks.layers.dmoe import ParallelDroplessMLP
from megablocks.layers.arguments import Arguments
from megablocks import ops
from functools import partial
import torch
from scattermoe.mlp import MLP

from torch import nn
import numpy as np
import pickle

import gc

def test_strategy(fwd, bwd):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    warmup = 50
    repetitions = 100
    fwd_timings = np.zeros(repetitions)
    bwd_timings = np.zeros(repetitions)
    for i in range(warmup):
        Y = fwd()
        bwd(Y)
    for rep in range(repetitions):
        starter.record()
        Y = fwd()
        ender.record()
        torch.cuda.synchronize()
        elapsed = starter.elapsed_time(ender)
        fwd_timings[rep] = elapsed
        starter.record()
        bwd(Y)
        ender.record()
        torch.cuda.synchronize()
        elapsed = starter.elapsed_time(ender)
        bwd_timings[rep] = elapsed
    return {"fwd": fwd_timings, "bwd": bwd_timings}

def baseline(L, xdim, hdim, dtype):
    X = torch.randn(L, xdim, dtype=dtype).cuda()
    DY = torch.randn(L, xdim, dtype=dtype).cuda()
    X.requires_grad_(True)
    dense_mlp = nn.Sequential(
        nn.Linear(xdim, hdim),
        nn.GELU(),
        nn.Linear(hdim, xdim)
    ).cuda().to(dtype)

    fwd = lambda: dense_mlp(X)
    bwd = lambda y: y.backward(DY)
    return {'dense': test_strategy(fwd, bwd)}

def init_scattermoe(xdim, hdim, E, k, dtype, X, DY, expert_p, expert_idxs):
    t_mlp = MLP(
        input_size=xdim, hidden_size=hdim, activation=nn.GELU(),
        num_experts=E, top_k=k).cuda().to(dtype)
    return (
        lambda: t_mlp(X, expert_p, expert_idxs),
        lambda y: y.backward(DY)
    )


def init_megablockmoe(xdim, hdim, E, k, dtype, X, DY, expert_p, expert_idxs):
    args = Arguments(
        hidden_size=xdim,
        ffn_hidden_size=hdim,
        moe_num_experts=E,
        moe_capacity_factor=1,
        moe_top_k=k,
        init_method=partial(torch.nn.init.normal_, mean=0.0, std=0.1),
        mlp_type='mlp',
        fp16=False,
        bf16=False,
        bias=False
    )
    mb_mlp = ParallelDroplessMLP(args).to(dtype)
    return (lambda: mb_mlp(X, expert_p, expert_p, expert_idxs),
            lambda y: y.backward(DY))

method_inits = {
    "simple": init_scattermoe,
    "megablocks": init_megablockmoe,
}

def test_params(init_fun, L, xdim, hdim, E, k, dtype):
    X = torch.randn(L, xdim, dtype=dtype).cuda()
    logits = torch.randn(L, E, dtype=dtype).cuda()
    DY = torch.randn(L, xdim, dtype=dtype).cuda()
    X.requires_grad_(True)
    expert_p, expert_idxs = torch.topk(logits, k=k)
    expert_p.requires_grad_(True)
    fwd, bwd = init_fun(xdim, hdim, E, k, dtype, X, DY, expert_p, expert_idxs)
    return test_strategy(fwd, bwd)
    
def batch_benchmarks():
    E = 64
    k = 8
    xdim = 4096
    results = {}
    print(method_inits.keys())
    for b in range(2, 30 + 2, 2):
    # for b in [16]:
        L = 2048 * b
        hdim = (2 * xdim) // k
        results[L] = {}
        results[L]['active_dense'] = baseline(L, xdim, k * hdim, dtype)['dense']
        results[L]['total_dense'] = baseline(L, xdim, E * hdim, dtype)['dense']
        print(L, end=' ')
        for m in method_inits:
            results[L][m] = test_params(method_inits[m], L, xdim, hdim, E, k, dtype)
            total_time = sum(results[L][m][p] for p in ['fwd', 'bwd'])
            print(total_time.mean(), end=' ')
        print()
    pickle.dump(results, open('batch_benchmarks.pkl', 'wb'))



def k_benchmarks():
    # Fix intermediate state and change k
    expansion_factor = 2
    sparse_factor = 8
    L = 2048 * 8
    xdim = 4096

    intermediate = xdim * expansion_factor

    small_result = baseline(L, xdim, intermediate, dtype)
    large_result = baseline(L, xdim, sparse_factor * intermediate, dtype)
    results = {}
    print(method_inits.keys()) 
    for k in [1, 2, 4, 8, 16]:
        hdim = intermediate // k
        E = sparse_factor * k
        assert intermediate % k == 0

        results[k] = {}
        for m in method_inits:
            results[k][m] = test_params(method_inits[m], L, xdim, hdim, E, k, dtype)

        # total_time_simple = sum(results[k]['simple'][p] for p in ['fwd', 'bwd'])
        # total_time_megablocks = sum(results[k]['megablocks'][p] for p in ['fwd', 'bwd'])
        print(xdim, hdim, E, k, end=' ')
        for method in method_inits:
            total_time = sum(results[k][method][p] for p in ['fwd', 'bwd'])
            print(total_time.mean(), end=' ')
        print()
    pickle.dump((small_result, large_result, results), open('k_benchmarks.pkl', 'wb'))





if __name__ == "__main__":
    dtype = torch.bfloat16

    print("Granularity benchmarks:")
    # k_benchmarks()
    print("Batch size benchmarks:")
    batch_benchmarks()


    L = 2048 * 8
    xdim = 4096
    E = 64
    k = 8
    hdim = 1024
    X = torch.randn(L, xdim, dtype=dtype).cuda()
    logits = torch.randn(L, E, dtype=dtype).cuda()
    DY = torch.randn(L, xdim, dtype=dtype).cuda()
    X.requires_grad_(True)
    expert_p, expert_idxs = torch.topk(logits, k=k)
    expert_p.requires_grad_(True)

    print("Megablocks MoE memory profiling...")
    args = Arguments(
        hidden_size=xdim,
        ffn_hidden_size=hdim,
        moe_num_experts=E,
        moe_capacity_factor=1,
        moe_top_k=k,
        init_method=partial(torch.nn.init.normal_, mean=0.0, std=0.1),
        mlp_type='mlp',
        fp16=False,
        bf16=False,
        bias=False
    )

    mb_mlp = ParallelDroplessMLP(args).to(dtype)
    torch.cuda.memory._record_memory_history()
    with torch.no_grad():
        Y = mb_mlp(X, logits, expert_p, expert_idxs)
    # Y.backward(DY)
    torch.cuda.memory._dump_snapshot("megablocks_memory.pkl")

    print("ScatterMoE memory profiling...")
    t_mlp = MLP(
        input_size=xdim, hidden_size=hdim, activation=nn.GELU(),
        num_experts=E, top_k=k).cuda().to(dtype)
    torch.cuda.memory._record_memory_history()
    Y = t_mlp(X, expert_p, expert_idxs)
    Y.backward(DY)
    torch.cuda.memory._dump_snapshot("scattermoe_memory.pkl")


