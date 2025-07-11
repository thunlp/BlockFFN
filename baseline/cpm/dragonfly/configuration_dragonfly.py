import torch
from transformers.configuration_utils import PretrainedConfig


class DragonflyConfig(PretrainedConfig):
    model_type = "cpm_dragonfly"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_key_value_heads": "num_kv_heads",
        "hidden_act": "activate_fn",
        "hidden_size": "dim_model",
        "num_attention_heads": "num_heads",
        "intermediate_size": "dim_ff",
        "num_hidden_layers": "num_layers",
        "vocab_size": "vocab_size",
        "rms_norm_eps": "eps",
        "scale_emb": "scale_emb",
        "scale_depth": "scale_depth",
        "scale": "scale",
        "attention_scale": "attention_scale",
        "qk_norm": "qk_norm",
        "ffn_gated": "ffn_gated",
    }  # model specific to common

    def __init__(
        self,
        vocab_size=122753,  # TODO: do we need to change to 122880 = 960 * 128?
        dim_model=4096,
        num_heads=32,
        num_kv_heads=32,
        dim_head=128,
        dim_ff=11008,
        num_layers=32,
        dropout_p=0.0,
        activate_fn="silu",
        scale=False,
        scale_emb: float = 1.0,
        scale_depth: float = -1,
        dim_model_base: int = 256,
        eps=1e-5,
        init_std=0.02,
        dtype="bf16",
        base=10000,
        qk_norm=False,
        tie_lm_head=False,
        max_length=8192,
        pose_prob=0.0,
        pose_scaling_factor=1,
        rope_scaling_type="",
        rope_scaling_factor=1,
        orig_max_length=8192,
        tp=0,
        moe_num_experts=0,
        moe_top_k=2,
        moe_top_p=0.3,
        moe_routing_strategy="topk",
        num_shared_experts=0,
        disable_checkpoint=None,
        ffn_gated=True,
        attention_type="vanilla",
        dim_q_lora=0,
        dim_kv_lora=0,
        dim_q_nope_head=0,
        dim_q_pe_head=0,
        dim_v_head=0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.activate_fn = activate_fn
        self.scale = scale
        self.scale_emb = scale_emb
        self._dtype = dtype
        self.dim_model_base = dim_model_base
        self.scale_depth = scale_depth
        self.eps = eps
        self.init_std = init_std
        self.base = base
        self.qk_norm = qk_norm
        self.tie_lm_head = tie_lm_head
        self.use_bfloat16 = True if self._dtype == "bf16" else False
        self.pose_prob = pose_prob
        self.pose_scaling_factor = pose_scaling_factor
        self.rope_scaling_type = rope_scaling_type
        self.rope_scaling_factor = rope_scaling_factor
        self.max_length = max_length
        self.orig_max_length = orig_max_length
        self.tp = tp
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_top_p = moe_top_p
        self.moe_routing_strategy = moe_routing_strategy
        self.num_shared_experts=num_shared_experts
        self.disable_checkpoint = disable_checkpoint
        self.ffn_gated = ffn_gated

        # multi-latent attention
        self.attention_type = attention_type
        self.dim_q_lora = dim_q_lora
        self.dim_kv_lora = dim_kv_lora
        self.dim_q_nope_head = dim_q_nope_head
        self.dim_q_pe_head = dim_q_pe_head
        self.dim_v_head = dim_v_head

        super().__init__(architectures=["CPMDragonflyForCausalLM"])

    @property
    def scale_width(
        self,
    ):
        if self.scale:
            return self.dim_model / self.dim_model_base
        else:
            return 1.0

    @property
    def dtype(
        self,
    ):  # -> Any | None:
        if self._dtype == "bf16":
            return torch.bfloat16
        elif self._dtype == "fp16":
            return torch.half
        elif self._dtype == "float32":
            return torch.float
