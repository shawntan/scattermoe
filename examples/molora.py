from scattermoe.mlp import MLP
from torch import nn


if __name__ == "__main__":
    d = 1024
    rank = 128
    N = 16
    top_k = 2
    mixture_of_lora = MLP(
        input_size=d,
        hidden_size=rank,
        num_experts=N,
        top_k=top_k,
        activation=nn.Identity(),
    )
    print(mixture_of_lora)
    # MLP(
    #   k=2
    #   (experts): ParallelExperts(num_experts=16, input_size=1024, output_size=128)
    #   (output_experts): ParallelExperts(num_experts=16, input_size=128, output_size=1024)
    #   (activation): Identity()
    # )

