import torch
import transformers
import gc
from mixtral.modeling_mixtral import MixtralModel, MixtralForCausalLM
from mixtral.configuration_mixtral import MixtralConfig
MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"

if __name__ == "__main__":
    dtype = torch.bfloat16
    config = MixtralConfig.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True, torch_dtype=dtype)
    num_experts = config.num_local_experts
    print("Loading original...")
    model_orig = transformers.MixtralForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True)
    print("Initialising ScatterMoE")
    model = MixtralForCausalLM(config).to(dtype)
    state_dict_orig = model_orig.state_dict() 
    for n, p in model.named_parameters():
        assert p.dtype == torch.bfloat16
        if n in state_dict_orig:
            p.data[:] = state_dict_orig.pop(n)
        else:
            prefix, suffix = n.split('moe_mlp')
            for i in range(num_experts):
                if suffix == ".output_experts.weight":
                    w2_param_name = prefix + "experts.%d.w2.weight" % i
                    assert state_dict_orig[w2_param_name].dtype == torch.bfloat16
                    p.data[i, :, :] = state_dict_orig.pop(w2_param_name)
                else:
                    w1_param_name = prefix + "experts.%d.w1.weight" % i
                    w3_param_name = prefix + "experts.%d.w3.weight" % i
                    out_dim, in_dim = state_dict_orig[w1_param_name].size()
                    p.data[i, :out_dim, :] = state_dict_orig.pop(w3_param_name)
                    p.data[i, out_dim:, :] = state_dict_orig.pop(w1_param_name)
    assert len(state_dict_orig) == 0
    print("Saving to file.")
    model.to(dtype=torch.bfloat16).save_pretrained("./converted/", save_config=True)

