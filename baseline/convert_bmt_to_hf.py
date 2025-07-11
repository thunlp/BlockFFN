import os
import json
import torch
import argparse
from collections import OrderedDict
from bmt_to_hf.configuration_minicpm import MiniCPMConfig
from bmt_to_hf.modeling_minicpm import MiniCPMForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
bmt:
encoder.layers.0.self_att.layernorm_before_attention.weight torch.Size([768])
encoder.layers.0.self_att.self_attention.project_q.weight torch.Size([768, 768])
encoder.layers.0.self_att.self_attention.project_k.weight torch.Size([768, 768])
encoder.layers.0.self_att.self_attention.project_v.weight torch.Size([768, 768])
encoder.layers.0.self_att.self_attention.attention_out.weight torch.Size([768, 768])
encoder.layers.0.ffn.layernorm_before_ffn.weight torch.Size([768])
encoder.layers.0.ffn.ffn.w_in.w_0.weight torch.Size([1920, 768])
encoder.layers.0.ffn.ffn.w_in.w_1.weight torch.Size([1920, 768])
encoder.layers.0.ffn.ffn.w_out.weight torch.Size([768, 1920])
encoder.output_layernorm.weight torch.Size([768])
input_embedding.weight torch.Size([122753, 768])

==========

hf:
model.embed_tokens.weight torch.Size([122753, 768])
model.layers.0.self_attn.q_proj.weight torch.Size([768, 768])
model.layers.0.self_attn.k_proj.weight torch.Size([768, 768])
model.layers.0.self_attn.v_proj.weight torch.Size([768, 768])
model.layers.0.self_attn.o_proj.weight torch.Size([768, 768])
model.layers.0.mlp.gate_proj.weight torch.Size([1920, 768])
model.layers.0.mlp.up_proj.weight torch.Size([1920, 768])
model.layers.0.mlp.down_proj.weight torch.Size([768, 1920])
model.layers.0.input_layernorm.weight torch.Size([768])
model.layers.0.post_attention_layernorm.weight torch.Size([768])
model.norm.weight torch.Size([768])
lm_head.weight torch.Size([122753, 768])
"""

"""
For MLA parameters:
bmt:
encoder.layers.0.self_att.self_attention.q_down_proj.weight torch.Size([768, 1536])
encoder.layers.0.self_att.self_attention.q_down_layernorm.weight torch.Size([768])
encoder.layers.0.self_att.self_attention.q_up_proj.weight torch.Size([2304, 768])
encoder.layers.0.self_att.self_attention.kv_down_proj.weight torch.Size([320, 1536])
encoder.layers.0.self_att.self_attention.kv_down_layernorm.weight torch.Size([256])
encoder.layers.0.self_att.self_attention.kv_up_proj.weight torch.Size([3072, 256])
encoder.layers.0.self_att.self_attention.o_proj.weight torch.Size([1536, 1536])

==========

hf:
model.layers.0.self_attn.q_a_proj.weight torch.Size([768, 1536])
model.layers.0.self_attn.q_a_layernorm.weight torch.Size([768])
model.layers.0.self_attn.q_b_proj.weight torch.Size([2304, 768])
model.layers.0.self_attn.kv_a_proj_with_mqa.weight torch.Size([320, 1536])
model.layers.0.self_attn.kv_a_layernorm.weight torch.Size([256])
model.layers.0.self_attn.kv_b_proj.weight torch.Size([3072, 256])
model.layers.0.self_attn.o_proj.weight torch.Size([1536, 1536])
"""

"""
For BlockFFN parameters:
bmt:
encoder.layers.0.ffn.ffn.moe_router.w_gate torch.Size([1280, 40])
encoder.layers.0.ffn.ffn.moe_experts.moe_w_in torch.Size([40, 128, 1280])
encoder.layers.0.ffn.ffn.moe_experts.moe_w_out torch.Size([40, 1280, 128])
encoder.layers.0.ffn.ffn.router_norm.weight torch.Size([40])

hf:
model.layers.0.mlp.up_proj torch.Size([40, 128, 1280])
model.layers.0.mlp.down_proj torch.Size([40, 1280, 128])
model.layers.0.mlp.router_proj.weight torch.Size([40, 1280])
model.layers.0.mlp.router_norm.weight torch.Size([40])
"""

"""
For MoE parameters:
bmt:
encoder.layers.0.ffn.ffn.experts.w_in.w_0.weight torch.Size([40, 128, 1280])
encoder.layers.0.ffn.ffn.experts.w_out.weight torch.Size([40, 1280, 128])
encoder.layers.0.ffn.ffn.router.weight torch.Size([40, 1280])

hf:
model.layers.0.mlp.experts.up_proj.weight torch.Size([40, 128, 1280])
model.layers.0.mlp.experts.down_proj.weight torch.Size([40, 1280, 128])
model.layers.0.mlp.router.weight torch.Size([40, 1280])
"""


def get_hf_config(path: str):
    if os.path.exists(f"{path}/config.json"):
        with open(f"{path}/config.json") as fin:
            tmp = json.load(fin)
        if "dim_model" in tmp:
            os.system(f"mv {path}/config.json {path}/config_bmt.json")
    assert os.system(f"cp bmt_to_hf/*.json {path}/") == 0
    assert os.system(f"cp bmt_to_hf/*.py {path}/") == 0

    with open(f"{path}/config_bmt.json") as fin:
        bmt_config: dict = json.load(fin)
    with open("bmt_to_hf/config.json") as fin:
        hf_config: dict = json.load(fin)

    # for general settings
    hf_config.update({
        "hidden_size": bmt_config["dim_model"],
        "num_attention_heads": bmt_config["num_heads"],
        "num_hidden_layers": bmt_config["num_layers"],
        "vocab_size": bmt_config["vocab_size"],
        "dim_model_base": bmt_config["dim_model_base"],
        "scale_emb": bmt_config["scale_emb"],
        "scale_depth": bmt_config["scale_depth"],
        "tie_word_embeddings": bmt_config["tie_lm_head"],
        "norm_after_router": bmt_config.get("norm_after_router", "none"),
        "norm_scale": bmt_config.get("norm_scale", 1.0),
    })
    # for attention
    is_mla = "dim_q_lora" in bmt_config and bmt_config["dim_q_lora"] > 0
    if is_mla:
        hf_config.update({
            "attention_type": "mla",
            "q_lora_rank": bmt_config["dim_q_lora"],
            "kv_lora_rank": bmt_config["dim_kv_lora"],
            "qk_nope_head_dim": bmt_config["dim_q_nope_head"],
            "qk_rope_head_dim": bmt_config["dim_q_pe_head"],
            "v_head_dim": bmt_config["dim_v_head"] if "dim_v_head" in bmt_config else bmt_config["dim_q_nope_head"],
            "num_key_value_heads": bmt_config["num_kv_heads"] if "num_kv_heads" in bmt_config else bmt_config["num_heads"],
        })
    else:
        hf_config.update({
            "attention_type": "vanilla",
            "num_key_value_heads": bmt_config["num_kv_heads"],
        })
    # for ffn
    if "ffn_type" in bmt_config and bmt_config["ffn_type"] in ["block", "block_linear"]:
        ffn_type = bmt_config["ffn_type"]
    elif "moe_routing_strategy" in bmt_config:
        ffn_type = "moe"
    else:
        ffn_type = "vanilla"
    hf_config["ffn_type"] = ffn_type
    hf_config["ffn_gated"] = bmt_config["ffn_gated"]
    if ffn_type in ["block", "block_linear"]:
        hf_config.update({
            "hidden_act": bmt_config["ffn_activate_fn"],
            "router_act": bmt_config["router_activate_fn"],
            "expert_size": bmt_config["dim_expert"],
            "num_experts": bmt_config["num_expert"],
            "block_implementation": bmt_config.get("block_implementation", "torch"),
        })
    elif ffn_type == "moe":
        hf_config.update({
            "hidden_act": bmt_config["activate_fn"],
            "intermediate_size": bmt_config["dim_ff"],
            "num_experts": bmt_config["moe_num_experts"],
            "moe_routing_strategy": bmt_config["moe_routing_strategy"],
            "num_shared_experts": bmt_config.get("num_shared_experts", 0),
            "moe_top_k": bmt_config["moe_top_k"],
            "moe_top_p": bmt_config.get("moe_top_p", -1),
        })
    else:
        hf_config.update({
            "hidden_act": bmt_config["activate_fn"] if "activate_fn" in bmt_config else bmt_config["ffn_activate_fn"],
            "intermediate_size": bmt_config["dim_ff"],
        })

    with open(f"{path}/config.json", "w") as f:
        json.dump(hf_config, f, indent=4, separators=(",", ": "))
    return MiniCPMConfig(**hf_config)


def inspect_state_keys(states):
    for key, val in states.items():
        tokens = key.split(".")
        if all(not token.isdigit() for token in tokens) or "0" in tokens:
            print(key, val.shape)


def check_converted_model(path: str):
    #try:
    #    tokenizer = AutoTokenizer.from_pretrained(path)
    #except TypeError:
    #    tokenizer = LlamaTokenizer(vocab_file="/home/test/test06/lyq/checkpoints/0.1b-2048_relu_fit-fix/130000/tokenizer.model")
    #    tokenizer.save_pretrained(path)

    assert os.system(f"cp /home/test/test06/lyq/old_checkpoints/sparsity-scaling/0.1b_relu/hf_model/tokenizer_config.json {path}/") == 0
    assert os.system(f"cp /home/test/test06/lyq/old_checkpoints/sparsity-scaling/0.1b_relu/hf_model/tokenizer.model {path}/") == 0
    assert os.system(f"cp /home/test/test06/lyq/old_checkpoints/sparsity-scaling/0.1b_relu/hf_model/special_tokens_map.json {path}/") == 0
    tokenizer = AutoTokenizer.from_pretrained(path)

    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    prompt = "Tsinghua Univerisity is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    inputs["max_length"] = 100
    outputs = model.generate(**inputs)
    print("OUT:", tokenizer.decode(outputs[0]))


def convert_bmt_to_hf():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.path):
        AutoTokenizer.from_pretrained(args.path)
        AutoModelForCausalLM.from_pretrained(args.path, trust_remote_code=True)
        return

    pt_path = [file for file in os.listdir(args.path) if file.endswith(".pt")]
    assert len(pt_path) == 1, str(pt_path)
    pt_path = os.path.join(args.path, pt_path[0])
    bmt_states = torch.load(pt_path, map_location="cpu")

    inspect_state_keys(bmt_states)

    hf_config = get_hf_config(args.path)
    hf_model = MiniCPMForCausalLM(hf_config)
    print("=" * 10)
    inspect_state_keys(hf_model.state_dict())

    hf_states = OrderedDict()
    hf_states["model.embed_tokens.weight"] = bmt_states["input_embedding.weight"]
    hf_states["model.norm.weight"] = bmt_states["encoder.output_layernorm.weight"]
    hf_states["lm_head.weight"] = bmt_states["input_embedding.weight"] if hf_config.tie_word_embeddings \
        else bmt_states["lm_head.weight"]
    for lid in range(hf_config.num_hidden_layers):
        if hf_config.attention_type == "vanilla":
            hf_states[f"model.layers.{lid}.self_attn.q_proj.weight"] = \
                bmt_states[f"encoder.layers.{lid}.self_att.self_attention.project_q.weight"]
            hf_states[f"model.layers.{lid}.self_attn.k_proj.weight"] = \
                bmt_states[f"encoder.layers.{lid}.self_att.self_attention.project_k.weight"]
            hf_states[f"model.layers.{lid}.self_attn.v_proj.weight"] = \
                bmt_states[f"encoder.layers.{lid}.self_att.self_attention.project_v.weight"]
            hf_states[f"model.layers.{lid}.self_attn.o_proj.weight"] = \
                bmt_states[f"encoder.layers.{lid}.self_att.self_attention.attention_out.weight"]
        elif hf_config.attention_type == "mla":
            hf_states[f"model.layers.{lid}.self_attn.q_a_proj.weight"] = \
                bmt_states[f"encoder.layers.{lid}.self_att.self_attention.q_down_proj.weight"]
            hf_states[f"model.layers.{lid}.self_attn.q_a_layernorm.weight"] = \
                bmt_states[f"encoder.layers.{lid}.self_att.self_attention.q_down_layernorm.weight"]
            hf_states[f"model.layers.{lid}.self_attn.q_b_proj.weight"] = \
                bmt_states[f"encoder.layers.{lid}.self_att.self_attention.q_up_proj.weight"]
            hf_states[f"model.layers.{lid}.self_attn.kv_a_proj_with_mqa.weight"] = \
                bmt_states[f"encoder.layers.{lid}.self_att.self_attention.kv_down_proj.weight"]
            hf_states[f"model.layers.{lid}.self_attn.kv_a_layernorm.weight"] = \
                bmt_states[f"encoder.layers.{lid}.self_att.self_attention.kv_down_layernorm.weight"]
            hf_states[f"model.layers.{lid}.self_attn.kv_b_proj.weight"] = \
                bmt_states[f"encoder.layers.{lid}.self_att.self_attention.kv_up_proj.weight"]
            hf_states[f"model.layers.{lid}.self_attn.o_proj.weight"] = \
                bmt_states[f"encoder.layers.{lid}.self_att.self_attention.o_proj.weight"]
        else:
            raise NotImplementedError()

        if hf_config.ffn_type == "vanilla":
            hf_states[f"model.layers.{lid}.mlp.gate_proj.weight"] = bmt_states[f"encoder.layers.{lid}.ffn.ffn.w_in.w_0.weight"]
            hf_states[f"model.layers.{lid}.mlp.up_proj.weight"] = bmt_states[f"encoder.layers.{lid}.ffn.ffn.w_in.w_1.weight"]
            hf_states[f"model.layers.{lid}.mlp.down_proj.weight"] = bmt_states[f"encoder.layers.{lid}.ffn.ffn.w_out.weight"]
        elif hf_config.ffn_type in ["block", "block_linear"]:
            if hf_config.block_implementation in ["torch", "kernel"]:
                hf_states[f"model.layers.{lid}.mlp.up_proj"] = bmt_states[f"encoder.layers.{lid}.ffn.ffn.moe_experts.moe_w_in"]
                hf_states[f"model.layers.{lid}.mlp.down_proj"] = bmt_states[f"encoder.layers.{lid}.ffn.ffn.moe_experts.moe_w_out"]
            elif hf_config.block_implementation == "fastmoe":
                hf_states[f"model.layers.{lid}.mlp.experts.up_proj.weight"] = bmt_states[f"encoder.layers.{lid}.ffn.ffn.moe_experts.moe_w_in"].view(hf_config.num_experts, hf_config.expert_size, hf_config.hidden_size).contiguous()
                hf_states[f"model.layers.{lid}.mlp.experts.down_proj.weight"] = bmt_states[f"encoder.layers.{lid}.ffn.ffn.moe_experts.moe_w_out"].view(hf_config.hidden_size, hf_config.num_experts, hf_config.expert_size).transpose(0, 1).contiguous()
            elif hf_config.block_implementation == "megablocks":
                hf_states[f"model.layers.{lid}.mlp.mlp.up_proj"] = bmt_states[f"encoder.layers.{lid}.ffn.ffn.moe_experts.moe_w_in"].contiguous()
                hf_states[f"model.layers.{lid}.mlp.mlp.down_proj"] = bmt_states[f"encoder.layers.{lid}.ffn.ffn.moe_experts.moe_w_out"].transpose(0, 1).contiguous()
            else:
                raise NotImplementedError()

            router_weight = bmt_states[f"encoder.layers.{lid}.ffn.ffn.moe_router.w_gate"]
            if router_weight.shape[0] == hf_config.hidden_size:
                hf_states[f"model.layers.{lid}.mlp.router_proj.weight"] = router_weight.T
            else:
                hf_states[f"model.layers.{lid}.mlp.router_proj.weight"] = router_weight
            if hf_config.norm_after_router == "rms":
                hf_states[f"model.layers.{lid}.mlp.router_norm.weight"] = bmt_states[f"encoder.layers.{lid}.ffn.ffn.router_norm.weight"]
        elif hf_config.ffn_type == "moe":
            hf_states[f"model.layers.{lid}.mlp.experts.up_proj.weight"] = bmt_states[f"encoder.layers.{lid}.ffn.ffn.experts.w_in.w_0.weight"]
            hf_states[f"model.layers.{lid}.mlp.experts.down_proj.weight"] = bmt_states[f"encoder.layers.{lid}.ffn.ffn.experts.w_out.weight"]
            hf_states[f"model.layers.{lid}.mlp.router.weight"] = bmt_states[f"encoder.layers.{lid}.ffn.ffn.router.weight"]
            if hf_config.num_shared_experts > 0:
                hf_states[f"model.layers.{lid}.mlp.shared_experts.up_proj.weight"] = bmt_states[f"encoder.layers.{lid}.ffn.ffn.shared_experts.w_in.w_0.weight"]
                hf_states[f"model.layers.{lid}.mlp.shared_experts.down_proj.weight"] = bmt_states[f"encoder.layers.{lid}.ffn.ffn.shared_experts.w_out.weight"]
        else:
            raise NotImplementedError()

        hf_states[f"model.layers.{lid}.input_layernorm.weight"] = \
            bmt_states[f"encoder.layers.{lid}.self_att.layernorm_before_attention.weight"]
        hf_states[f"model.layers.{lid}.post_attention_layernorm.weight"] = \
            bmt_states[f"encoder.layers.{lid}.ffn.layernorm_before_ffn.weight"]
    torch.save(hf_states, f"{args.path}/pytorch_model.bin")
    check_converted_model(args.path)


if __name__ == "__main__":
    convert_bmt_to_hf()
