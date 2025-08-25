import pytest
import torch
from torch import nn
from torch.nn import functional as F
from scattermoe.mlp import MLP
import scattermoe

scattermoe.kernels.ops.ALLOW_TF32 = False

def dumb_forward(m, x, expert_p, expert_idxs):
    output = torch.stack([
        sum(
            expert_p[i, j] * F.linear(
                m.activation(
                    F.linear(
                        x[i], m.experts.weight[expert_idxs[i, j]],
                        bias=m.experts.bias[expert_idxs[i, j]] if m.experts.bias is not None else None
                    )
                ),
                m.output_experts.weight[expert_idxs[i, j]],
                bias=m.output_experts.bias[expert_idxs[i, j]] if m.output_experts.bias is not None else None
            )
            for j in range(expert_idxs.size(1))
        ) for i in range(expert_idxs.size(0))
    ], dim=0)
    return output

def assert_diff(name, ref_X, new_X, tolerance=1e-2):
    diff = torch.abs(ref_X - new_X)
    max_diff = diff.max()
    print(f"{name} diff: {max_diff.item()}")
    assert max_diff < tolerance


class TestClass:
    @pytest.mark.parametrize('dtype', [torch.float32])
    @pytest.mark.parametrize('bias', [False, True])
    @pytest.mark.parametrize('length', [1, 256, 512])
    @pytest.mark.parametrize('E', [8])
    @pytest.mark.parametrize('x_dim, h_dim, k', [
        (xd, (4 * xd) // k, k)
        for xd in [128, 256, 512, 600, 100]
        for k in [2, 3, 4]
    ])
    def test_mlp_correctness(self, length, x_dim, h_dim, E, k, bias, dtype):
        logits = torch.randn(length, E, dtype=dtype)
        weights = torch.softmax(logits.float(), axis=-1).cuda().to(dtype)
        X = torch.randn(length, x_dim, dtype=dtype, requires_grad=True).cuda()
        DY = torch.randn(length, x_dim, dtype=dtype).cuda()
        k_weights, k_idxs = torch.topk(weights, k)
        k_weights.requires_grad_()

        mlp = MLP(
            input_size=x_dim, hidden_size=h_dim,
            activation=nn.GELU(),
            num_experts=E, top_k=k,
            bias=bias
        ).cuda().to(dtype)
        if bias:
            nn.init.normal_(mlp.experts.bias, std=0.02)
            nn.init.normal_(mlp.output_experts.bias, std=0.02)
            


        Y = mlp(X, k_weights, k_idxs)
        name_tup = ("dX", "dg", "dW1", "dW2")
        input_tup = (X, k_weights, mlp.experts.weight, mlp.output_experts.weight)
        if bias:
            name_tup += ("db1", "db2")
            input_tup += (mlp.experts.bias, mlp.output_experts.bias)

        ref_out_tup = torch.autograd.grad(
            outputs=(Y,),
            inputs=input_tup,
            grad_outputs=(DY,)
        )
        Y_ = dumb_forward(mlp, X, k_weights, k_idxs)
        out_tup = torch.autograd.grad(
            outputs=(Y_,),
            inputs=input_tup,
            grad_outputs=(DY,)
        )
        tol = 1e-4 if dtype == torch.float32 else 1e-2

        assert_diff("Y", Y_, Y, tolerance=tol)
        for name, ref, new in zip(name_tup, ref_out_tup, out_tup):
            assert_diff(name, ref, new, tolerance=tol)