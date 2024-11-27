import pytest
import torch
from torch import nn
from torch.nn import functional as F
from scattermoe.mlp import MLP


def dumb_forward(m, x, expert_p, expert_idxs):
    output = torch.stack([
        sum(
            expert_p[i, j] * F.linear(
                m.activation(F.linear(x[i], m.experts.weight[expert_idxs[i, j]])),
                m.output_experts.weight[expert_idxs[i, j]]
            )
            for j in range(expert_idxs.size(1))
        ) for i in range(expert_idxs.size(0))
    ], dim=0)
    return output

class TestClass:
    @pytest.mark.parametrize('dtype', [torch.float32])
    @pytest.mark.parametrize('E', [8, 4, 16, 32])
    @pytest.mark.parametrize('x_dim, h_dim, k', [
        (xd, (4 * xd) // k, k)
        for xd in [512, 1024, 2048]
        for k in [2, 3, 4]
    ])
    @pytest.mark.parametrize('length', [1, 256, 512, 1024, 2048, 4096][::-1])
    def test_mlp_correctness(self, length, x_dim, h_dim, E, k, dtype):
        logits = torch.randn(length, E, dtype=dtype)
        weights = torch.softmax(logits.float(), axis=-1).cuda().to(dtype)
        X = torch.randn(length, x_dim, dtype=dtype, requires_grad=True).cuda()
        DY = torch.randn(length, x_dim, dtype=dtype).cuda()
        k_weights, k_idxs = torch.topk(weights, k)
        k_weights.requires_grad_()

        mlp = MLP(
            input_size=x_dim, hidden_size=h_dim,
            activation=nn.GELU(),
            num_experts=E, top_k=k
        ).cuda().to(dtype)


        Y = mlp(X, k_weights, k_idxs)
        dX, dg, dW1, dW2 = torch.autograd.grad(
            outputs=(Y,),
            inputs=(X, k_weights, mlp.experts.weight, mlp.output_experts.weight),
            grad_outputs=(DY,)
        )
        Y_ = dumb_forward(mlp, X, k_weights, k_idxs)
        dX_, dg_, dW1_, dW2_ = torch.autograd.grad(
            outputs=(Y_,),
            inputs=(X, k_weights, mlp.experts.weight, mlp.output_experts.weight),
            grad_outputs=(DY,)
        )
        err_Y = torch.abs(Y_ - Y)
        err_dX = torch.abs(dX_ - dX)
        err_dg = torch.abs(dg_ - dg)
        err_dW1 = torch.abs(dW1_ - dW1)
        err_dW2 = torch.abs(dW2_ - dW2)
        tolerance = 1e-2
        print()
        print("Y error:", err_Y.max())
        print("dg:", err_dg.max())
        print("dW1:", err_dW1.max())
        print("dW2:", err_dW2.max())
        print("dX:", err_dX.max())
        assert err_Y.max() < tolerance, "Y error too large: max %0.05f" % err_Y.max()
        assert err_dg.max() < tolerance, "dg error too large: max %0.05f" % err_dg.max()
        assert err_dW1.max() < tolerance, "dW1 error too large: max %0.05f" % err_dW1.max()
        assert err_dW2.max() < tolerance, "dW2 error too large: max %0.05f" % err_dW2.max()
        assert err_dX.max() < tolerance, "dX error too large: max %0.05f" % err_dX.max()

