# typing: strict
# coding=utf-8
# Copyright 2023 ModelBest Inc.

import math
from typing import Optional, Tuple, Union, Callable

import bmtrain as bmt
import torch
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_varlen_func

from .configuration_dragonfly import DragonflyConfig  # from cpm.utils import Config
from .moe_gates import TopKRouter, TopKExpertChoiceRouter, TopPRouter, ReLURouter
from .sparse_mixer import SparseMixerV2
from .moe_calculator import FastTopKCalculator, ExpertChoiceCalculator
from fmoe.linear import MOELinear
from .activation import ActivationContext
from .layer_norm import LayerNorm
from .activation_function import get_activation_fn

# TODO:
# 1. add scale_emb to embed and layernorm
# 2. add scale_width to all layers
# 3. add scale_depth to residual


class ScaledRotaryEmbeddingESM(bmt.DistributedModule):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    Add multiple Positional Interpolation methods:
    + [Linear](http://arxiv.org/abs/2306.15595)
    + [NTK-aware](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
    + [Dynamic Scaling](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/)
    + [NTK-by-parts](https://github.com/jquesnelle/yarn/pull/1)
    + [YaRN](http://arxiv.org/abs/2309.00071)
    Args:
        dim: Dimension of the input, attn_dim // n_heads.
        max_position_embeddings: Maximum number of positions to be embedded.
        base: Base of the positional encoding function.
        pose_prob: Probability of using PoSE.
        pose_scaling_factor: max_position_embeddings scaling factor for PoSE.
        scaling_type: Type of scaling to use, one of ["Linear", "NTK-aware", "Dynamic NTK", "NTK-by-parts", "YaRN", "Dynamic YaRN", ""].
        rope_scaling_factor: RoPE Scaling factor for scaling type, new max length / before extend max length.
        beta_fast: Number of rotations to use for fast angular velocity.
        beta_slow: Number of rotations to use for slow angular velocity.
        extrapolation_factor: [0, 1], 0 is fully extrapolation, 1 is fully NTK-by-parts/YaRN.
        attn_factor: Uniform attn scale factor for tuning YaRN, 1 is best for LLaMA-1/2.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        pose_prob: float = 0.0,
        pose_scaling_factor: float = 1.0,
        scaling_type: str = "",
        rope_scaling_factor: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        extrapolation_factor: int = 1,
        attn_factor: int = 1,
        original_max_position_embeddings: int = 2048,
        persistent: bool = True,
        dynamic_scaling_seq_len: int = 512,
        device=None,
    ):
        assert scaling_type in ["Linear", "NTK-aware", "Dynamic NTK", "NTK-by-parts", "YaRN", ""]
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.persistent = persistent
        self.device = device
        # scaling config
        self.scaling_type = scaling_type
        self.pose_scaling_factor = pose_scaling_factor
        self.rope_scaling_factor = rope_scaling_factor
        # PoSE
        self.pose_prob = pose_prob
        # NTK-by-parts and YaRN args
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.original_max_position_embeddings = original_max_position_embeddings

        if pose_prob > 0:
            self.scaled_max_position_embeddings = int(max_position_embeddings * pose_scaling_factor)
        else:
            self.scaled_max_position_embeddings = max_position_embeddings

        if self.scaling_type == "NTK-aware":
            base = self.base * (self.rope_scaling_factor ** (self.dim / (self.dim - 2)))
        else:
            base = self.base
        # TODO: Implement base NTK-aware in NTK-by-parts
        if self.scaling_type in ["NTK-by-parts", "YaRN"]:
            self._ntk_parts_update_inv_freq(self.scaled_max_position_embeddings)
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(self.device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Get n-d magnitude scaling corrected for interpolation
        self.m_scale = float(self._get_m_scale(self.rope_scaling_factor) * self.attn_factor)
        self._set_cos_sin_cache(dynamic_scaling_seq_len)

    def _get_m_scale(self, scale=1.0):
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def _ntk_parts_update_inv_freq(self, seq_len):
        # Inverse dim formula to find dim based on number of rotations
        def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
            return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

        # Find dim range bounds based on rotations
        def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
            low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
            high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))

            return max(low, 0), min(high, dim - 1)  # Clamp values just in case

        def linear_ramp_mask(min, max, dim):
            if min == max:
                max += 0.001  # Prevent singularity

            linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
            ramp_func = torch.clamp(linear_func, 0, 1)
            return ramp_func

        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(self.device) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.rope_scaling_factor * pos_freqs)
        low, high = find_correction_range(
            self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings
        )
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (
            1 - linear_ramp_mask(low, high, self.dim // 2).float().to(self.device)
        ) * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=self.persistent)

    def _set_cos_sin_cache(self, seq_len, device=None):
        self.max_seq_len_cached = seq_len
        if device is not None:
            self.device = device

        if self.scaling_type == "Dynamic NTK" and seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.rope_scaling_factor * seq_len / self.max_position_embeddings) - (self.rope_scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(self.device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=self.persistent)

        t = torch.arange(self.max_seq_len_cached, device=self.device).type_as(self.inv_freq)
        if self.scaling_type == "Linear":
            freqs = torch.outer(t / self.rope_scaling_factor, self.inv_freq.to(device=t.device).to(t.dtype))
        else:
            freqs = torch.outer(t, self.inv_freq.to(device=t.device).to(t.dtype))
        emb = torch.cat((freqs, freqs), dim=-1)
        if self.scaling_type == "YaRN":
            self.register_buffer("cos_cached", (emb.cos() * self.m_scale), persistent=self.persistent)
            self.register_buffer("sin_cached", (emb.sin() * self.m_scale), persistent=self.persistent)
        else:
            self.register_buffer("cos_cached", emb.cos(), persistent=self.persistent)
            self.register_buffer("sin_cached", emb.sin(), persistent=self.persistent)

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(self, q, k, cos, sin, position_ids) -> Tuple[torch.Tensor, torch.Tensor]:
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        orig_dtype = k.dtype
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        q_fp32 = q.to(dtype=torch.float32, device=q.device)
        k_fp32 = k.to(dtype=torch.float32, device=k.device)
        q_embed = (q_fp32 * cos) + (self._rotate_half(q_fp32) * sin)
        k_embed = (k_fp32 * cos) + (self._rotate_half(k_fp32) * sin)
        return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_dim, offset=0, cu_seqlens=None, max_length=None, position_ids=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_dim = (seq_dim + k.dim()) % k.dim()
        # get max current seq len from all workers
        if self.pose_prob > 0.0:
            seq_len = torch.max(position_ids) + 1
        else:
            seq_len = k.shape[seq_dim] + offset
        seq_len_tensor = torch.tensor(seq_len, device=self.device)
        seq_len_tensor_reduced = bmt.distributed.all_reduce(seq_len_tensor, op="max")
        seq_len_reduced = seq_len_tensor_reduced.item()
        # update cache if needed
        if seq_len_reduced > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        cos, sin = (
            self.cos_cached[:seq_len_reduced],
            self.sin_cached[:seq_len_reduced],
        )
        if position_ids.dtype != torch.long:  # 231108 input is int32
            position_ids = position_ids.to(dtype=torch.long)
        if cu_seqlens is None:
            q_embed, k_embed = self._apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        else:
            assert offset == 0, "past kv is not supported in flash attn"
            q_embed, k_embed = self._apply_rotary_pos_emb(q, k, cos, sin, position_ids.view(-1))

        return q_embed, k_embed


def Linear(*args, **kwargs):
    tp = kwargs.pop("tp", 0)
    num_experts = kwargs.get("num_experts", -1)
    if num_experts > 0:
        assert tp == 0
        return MoELinearExperts(*args, **kwargs)
    kwargs.pop("num_experts", 0)
    if tp == 0:
        return NormalLinear(*args, **kwargs)
    if tp == 1:
        return ColumnParallelLinear(*args, **kwargs)
    if tp == 2:
        return RowParallelLinear(*args, **kwargs)


class NormalLinear(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dtype: torch.dtype = torch.bfloat16,
        init_mean: float = 0.0,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out

        # TODO:init
        # init_std = 1 / ((dim_in + dim_out) ** 0.5)

        self.weight = bmt.DistributedParameter(
            torch.empty((dim_out, dim_in), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
        )

    def forward(self, x: torch.Tensor, fwd_expert_count: torch.Tensor = None):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501

        x = F.linear(x, self.weight, None)

        return x


class MoELinearExperts(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_experts: int,
        dtype: torch.dtype = torch.bfloat16,
        init_mean: float = 0.0,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out

        # TODO:init
        # init_std = 1 / ((dim_in + dim_out) ** 0.5)

        self.weight = bmt.DistributedParameter(
            torch.empty((num_experts, dim_out, dim_in), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
        )

    def forward(self, x: torch.Tensor, fwd_expert_count: torch.Tensor):
        x = MOELinear.apply(x, fwd_expert_count, self.weight, None)
        return x


class ColumnParallelLinear(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dtype: torch.dtype = torch.bfloat16,
        init_mean: float = 0.0,
        init_std: float = 0.02,
        gather_output=False,
        gather_input=True,
    ):
        super().__init__()
        assert dim_out % bmt.config["tp_size"] == 0

        # TODO: init
        # init_std = 1 / ((dim_in + dim_out) ** 0.5)

        dim_out = dim_out // bmt.config["tp_size"]
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.gather_input = gather_input
        self.gather_output = gather_output

        self.weight = bmt.DistributedParameter(
            torch.empty((dim_out, dim_in), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            tp_split_dim=0,
            tp_mode=True,
        )
        self.bias = None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501

        x = bmt.nn.OpParallelLinear.apply(
            x, self.weight, self.bias, self.gather_input, self.gather_output, False, None, 1
        )

        return x


class RowParallelLinear(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dtype: torch.dtype = torch.bfloat16,
        init_mean: float = 0.0,
        init_std: float = 0.02,
        split_input=False,
        all_reduce_output=False,
    ):
        super().__init__()
        assert dim_in % bmt.config["tp_size"] == 0
        # init_std = 1 / ((dim_in + dim_out) ** 0.5)

        dim_in = dim_in // bmt.config["tp_size"]
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out

        self.split_input = split_input
        self.all_reduce_output = all_reduce_output

        self.weight = bmt.DistributedParameter(
            torch.empty((dim_out, dim_in), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            tp_split_dim=1,
            tp_mode=True,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501
        if not self.all_reduce_output:
            x = x.view(x.shape[0] * bmt.config["tp_size"], -1, x.shape[-1])

        x = bmt.nn.OpParallelLinear.apply(
            x, self.weight, None, self.split_input, False, self.split_input, 1 if self.all_reduce_output else 2, 1
        )

        return x


class DenseGatedACT(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_ff: int,
        num_experts: int = -1,
        activate_fn: str = "silu",
        dtype=torch.bfloat16,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        _std = init_std / math.sqrt(scale_width) if scale else init_std

        self.w_0 = Linear(
            dim_in=dim_in,
            dim_out=dim_ff,
            dtype=dtype,
            tp=tp,
            init_std=_std,
            num_experts=num_experts,
        )

        self.w_1 = Linear(dim_in=dim_in, dim_out=dim_ff, dtype=dtype, tp=tp, init_std=_std, num_experts=num_experts)

        self.act = get_activation_fn(
            activate_fn, dim_norm=dim_ff, dtype=dtype, eps=eps,
        )
        bmt.print_rank(f"<<< DenseGatedACT.act {activate_fn} >>>")

    def forward(self, x: torch.Tensor, fwd_expert_count: torch.Tensor = None):
        """This model inherits from bmt.DistributedModule.
            Transform an input tensor from one feature space to another via a nonlinear operation

        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): Tensor that will be subject to nonlinear operations.

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_ff)``)

        """  # noqa: E501
        gate_score = self.act(self.w_0(x, fwd_expert_count))
        ActivationContext.stat_moe_intermediate_activation(gate_score)
        x = self.w_1(x, fwd_expert_count)

        x = gate_score * x
        return x


class DenseACT(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_ff: int,
        num_experts: int = -1,
        activate_fn: str = "silu",
        dtype=torch.bfloat16,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        _std = init_std / math.sqrt(scale_width) if scale else init_std

        self.w_0 = Linear(
            dim_in=dim_in,
            dim_out=dim_ff,
            dtype=dtype,
            tp=tp,
            init_std=_std,
            num_experts=num_experts,
        )

        self.act = get_activation_fn(
            activate_fn, dim_norm=dim_ff, dtype=dtype, eps=eps,
        )
        bmt.print_rank(f"<<< DenseACT.act {activate_fn} >>>")

    def forward(self, x: torch.Tensor, fwd_expert_count: torch.Tensor = None):
        """This model inherits from bmt.DistributedModule.
            Transform an input tensor from one feature space to another via a nonlinear operation

        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): Tensor that will be subject to nonlinear operations.

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_ff)``)

        """  # noqa: E501
        x = self.act(self.w_0(x, fwd_expert_count))
        ActivationContext.stat_moe_intermediate_activation(x)
        return x


class FeedForward(bmt.DistributedModule):
    r"""FeedForward module."""  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        num_experts: int = -1,
        activate_fn: str = "silu",
        dtype=torch.bfloat16,
        dropout_p: Optional[float] = None,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        ffn_gated: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()

        act_cls = DenseGatedACT if ffn_gated else DenseACT
        self.w_in = act_cls(
            dim_in=dim_model,
            dim_ff=dim_ff,
            activate_fn=activate_fn,
            dtype=dtype,
            scale=scale,
            init_std=init_std,
            scale_width=scale_width,
            num_experts=num_experts,
            eps=eps,
        )

        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        _std = init_std / math.sqrt(scale_width) if scale else init_std
        self.w_out = Linear(dim_in=dim_ff, dim_out=dim_model, dtype=dtype, init_std=_std, num_experts=num_experts)

    def forward(self, x: torch.Tensor, fwd_expert_count: torch.Tensor = None, seq_mask: torch.Tensor = None):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of feed-forward module.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of feed-forward module.
        """  # noqa: E501
        x = self.w_in(x, fwd_expert_count)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.w_out(x, fwd_expert_count)

        return x


class Embedding(bmt.DistributedModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        dtype: torch.dtype = torch.bfloat16,
        init_mean: float = 0.0,
        init_std: float = 1,
        scale: bool = False,
        scale_emb: float = 1.0,
        scale_width: float = 1.0,
        tp: int = 0,
    ):
        super().__init__()

        self.dim_model = embedding_size
        self.weight = bmt.DistributedParameter(
            torch.empty(vocab_size, embedding_size, dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
        )
        self.tp = tp
        self.scale = scale
        self.scale_emb = scale_emb
        self.scale_width = scale_width

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.
        Return:
            :obj:`torch.Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
        """  # noqa: E501

        if self.tp:
            x = x.view(-1).chunk(bmt.config["tp_size"])[bmt.config["tp_rank"]].view(x.size(0), -1)

        embeds = F.embedding(x, self.weight)

        if self.scale:
            embeds = embeds * self.scale_emb

        return embeds

    def projection(self, x: torch.Tensor):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection map embed_size back to vocab_size.
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, vocab_output_size)``: The projection output.
        """  # noqa: E501

        if self.scale:
            x = x / self.scale_width  # TODO: check if it is ok to add before all_gather

        logits = F.linear(x, self.weight)
        return logits


class Attention(bmt.DistributedModule):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        num_kv_heads: int,
        dim_head: int,
        dtype: torch.dtype = torch.bfloat16,
        dropout_p: Optional[float] = None,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_groups = num_heads // num_kv_heads
        self.dim_head = dim_head

        self.scale = scale
        _std = init_std / math.sqrt(scale_width) if scale else init_std

        self.project_q = Linear(
            self.dim_model,
            self.num_heads * self.dim_head,
            dtype=dtype,
            tp=tp,
            init_std=_std,
        )
        self.project_k = Linear(
            self.dim_model,
            self.num_kv_heads * self.dim_head,
            dtype=dtype,
            tp=tp,
            init_std=_std,
        )
        self.project_v = Linear(
            self.dim_model,
            self.num_kv_heads * self.dim_head,
            dtype=dtype,
            tp=tp,
            init_std=_std,
        )

        self.attention_out = Linear(
            self.num_heads * self.dim_head,
            self.dim_model,
            dtype=dtype,
            tp=tp * 2,  # TODO
            init_std=_std,
        )

        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(p=dropout_p)
            self.dropout_p = dropout_p
        else:
            self.dropout = None

        self.tp = tp

    def forward(
        self,
        hidden_q: torch.Tensor,
        hidden_kv: torch.Tensor,
        position_bias: Union[torch.Tensor, Callable],
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: int = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """This model inherits from bmt.DistributedModule."""  # noqa: E501

        batch_size = hidden_q.size(0)

        if self.tp:
            assert hidden_q.data_ptr() == hidden_kv.data_ptr()

            hidden_q = bmt.nn.OpParallelLinear.apply(
                hidden_q,
                torch.cat([self.project_q.weight, self.project_k.weight, self.project_v.weight], dim=0),
                torch.cat([self.project_q.bias, self.project_k.bias, self.project_v.bias], dim=0)
                if self.project_q.bias is not None
                else None,
                True,
                False,
                False,
                None,
                1,
            )

            hidden_q = hidden_q.view(batch_size, -1, hidden_q.shape[-1])

            block_size = hidden_q.shape[-1] // (self.head_groups + 1 + 1)
            h_q = hidden_q[..., : block_size * self.head_groups]
            h_k = hidden_q[..., block_size * self.head_groups : block_size * (self.head_groups + 1)]
            h_v = hidden_q[..., block_size * (self.head_groups + 1) :]
        else:
            h_q = self.project_q(hidden_q)
            h_k = self.project_k(hidden_kv)
            h_v = self.project_v(hidden_kv)

        len_q = h_q.size(1)
        len_k = h_k.size(1)

        h_q = h_q.view(batch_size * len_q, -1, self.dim_head)
        h_k = h_k.view(batch_size * len_k, -1, self.dim_head)
        h_v = h_v.view(batch_size * len_k, -1, self.dim_head)
        h_q, h_k = position_bias(h_q, h_k, -3, cu_seqlens=cu_seqlens, max_length=max_seqlen, position_ids=position_ids)
        score = flash_attn_varlen_func(
            h_q,
            h_k,
            h_v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            self.dropout_p,
            causal=True,
        )

        score = score.view(batch_size, len_q, -1)
        score = self.attention_out(score)

        return score


class MultiLatentAttention(bmt.DistributedModule):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_q_nope_head: int,
        dim_q_pe_head: int,
        dim_v_head: int,
        dim_q_lora: int,
        dim_kv_lora: int,
        dtype: torch.dtype = torch.bfloat16,
        dropout_p: Optional[float] = None,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        qk_norm: bool = False,
        layer_id: int = 0,
    ) -> None:
        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_q_nope_head = dim_q_nope_head
        self.dim_q_pe_head = dim_q_pe_head
        self.dim_q_lora = dim_q_lora
        self.dim_kv_lora = dim_kv_lora
        self.dim_q_head = self.dim_q_pe_head + self.dim_q_nope_head
        self.dim_v_head = dim_v_head

        self.scale = scale
        _std = init_std / math.sqrt(scale_width) if scale else init_std

        if self.dim_q_lora is None:
            self.q = Linear(self.dim_model, self.num_heads * self.dim_q_head, dtype=dtype, tp=tp, init_std=_std)
        else:
            # W_DQ
            self.q_down_proj = Linear(self.dim_model, self.dim_q_lora, dtype=dtype, tp=tp, init_std=_std)
            self.q_down_layernorm = LayerNorm(dim_norm=self.dim_q_lora)
            # [W_UQ; W_QR]
            self.q_up_proj = Linear(self.dim_q_lora, self.num_heads * self.dim_q_head, dtype=dtype, tp=tp, init_std=_std)

        # [W_DKV; W_KR]
        self.kv_down_proj = Linear(self.dim_model, self.dim_kv_lora + self.dim_q_pe_head, dtype=dtype, tp=tp, init_std=_std)
        self.kv_down_layernorm = LayerNorm(dim_norm=self.dim_kv_lora)
        # [W_UK; W_UV]
        self.kv_up_proj = Linear(self.dim_kv_lora, self.num_heads * (self.dim_q_head - self.dim_q_pe_head + self.dim_v_head))

        # W_O
        self.o_proj = Linear(self.num_heads * self.dim_v_head, self.dim_model, dtype=dtype, tp=tp, init_std=_std)

        self.tp = tp
        self.dropout_p = dropout_p
        self.layer_id = layer_id

    def forward(
        self,
        hidden_q: torch.Tensor,
        hidden_kv: torch.Tensor,
        position_bias: Union[torch.Tensor, Callable],
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: int = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """This model inherits from bmt.DistributedModule"""  # noqa: E501

        bsz, len_q, _ = hidden_q.size()

        assert self.tp == 0

        if self.dim_q_lora is None:
            q = self.q(hidden_q)
        else:
            q = self.q_up_proj(self.q_down_layernorm(self.q_down_proj(hidden_q)))
        q = q.view(bsz, len_q, self.num_heads, self.dim_q_head)
        q_nope, q_pe = torch.split(
            q, [self.dim_q_nope_head, self.dim_q_pe_head], dim=-1
        )

        compr_kv = self.kv_down_proj(hidden_kv)
        compr_kv, k_pe = torch.split(
            compr_kv, [self.dim_kv_lora, self.dim_q_pe_head], dim=-1
        )
        k_pe = k_pe.unsqueeze(dim=-2)

        kv = (
            self.kv_up_proj(self.kv_down_layernorm(compr_kv))
            .view(bsz, len_q, self.num_heads, self.dim_q_nope_head + self.dim_v_head)
        )
        k_nope, h_v = torch.split(kv, [self.dim_q_nope_head, self.dim_v_head], dim=-1)
        h_q_pe, h_k_pe = position_bias(q_pe, k_pe, -3, cu_seqlens=cu_seqlens, max_length=max_seqlen, position_ids=position_ids)

        h_q = torch.empty((bsz, len_q, self.num_heads, self.dim_q_head), dtype=torch.bfloat16, device=hidden_q.device)
        h_q[:, :, :, :self.dim_q_nope_head] = q_nope
        h_q[:, :, :, self.dim_q_nope_head:] = h_q_pe

        # copy h_k_pe self.num_heads times, shared key with position info
        h_k = torch.empty((bsz, len_q, self.num_heads, self.dim_q_head), dtype=torch.bfloat16, device=hidden_kv.device)
        h_k[:, :, :, :self.dim_q_nope_head] = k_nope
        h_k[:, :, :, self.dim_q_nope_head:] = h_k_pe

        h_q = h_q.view(bsz * len_q, self.num_heads, self.dim_q_head)
        h_k = h_k.view(bsz * len_q, self.num_heads, self.dim_q_head)
        h_v = h_v.view(bsz * len_q, self.num_heads, self.dim_v_head)

        # fit in fa
        if self.dim_q_head != self.dim_v_head:
            h_v = torch.nn.functional.pad(h_v, [0, self.dim_q_head - self.dim_v_head])

        score = flash_attn_varlen_func(
            h_q,
            h_k,
            h_v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            self.dropout_p,
            causal=True,
            deterministic=True,
        )
        if self.dim_q_head != self.dim_v_head:
            score = score[:, :, :self.dim_v_head]

        score = score.reshape(bsz, len_q, -1)
        score = self.o_proj(score)
        return score


class SelfAttentionBlock(bmt.DistributedModule):
    """The whole cross-attention block. A sequence of operation. Consists of layernorm, self-attention and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.bfloat16.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        num_kv_heads: int,
        dim_head: int,
        dtype=torch.bfloat16,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        scale_depth: float = -1,
        qk_norm: bool = False,
        layer_id: int = 0,
        num_layers: int = 0,
        attention_type: str = "vanilla",
        dim_q_lora: int = 0,
        dim_kv_lora: int = 0,
        dim_q_nope_head: int = 0,
        dim_q_pe_head: int = 0,
        dim_v_head: int = 0,
    ):
        super().__init__()

        self.layernorm_before_attention = LayerNorm(
            dim_model,
            dtype=dtype,
            eps=eps,
        )

        if attention_type == "vanilla":
            self.self_attention = Attention(
                dim_model=dim_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dim_head=dim_head,
                dtype=dtype,
                dropout_p=dropout_p,
                tp=tp,
                scale=scale,
                init_std=init_std,
                scale_width=scale_width,
                qk_norm=qk_norm,
            )
        elif attention_type == "mla":
            self.self_attention = MultiLatentAttention(
                dim_model=dim_model,
                num_heads=num_heads,
                dim_q_nope_head=dim_q_nope_head,
                dim_q_pe_head=dim_q_pe_head,
                dim_v_head=dim_v_head,
                dim_q_lora=dim_q_lora,
                dim_kv_lora=dim_kv_lora,
                dtype=dtype,
                dropout_p=dropout_p,
                tp=tp,
                scale=scale,
                init_std=init_std,
                scale_width=scale_width,
                qk_norm=qk_norm,
                layer_id=layer_id,
            )
        else:
            raise NotImplementedError(f"invalid attention_type {attention_type}")

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.scale = scale
        self.scale_depth = scale_depth
        self.num_layers = num_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: ScaledRotaryEmbeddingESM,
        cu_seqlens: torch.Tensor,
        max_seqlen: int = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        x = self.layernorm_before_attention(hidden_states)
        x = self.self_attention(
            x,
            x,
            position_bias,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids,
        )

        if self.dropout is not None:
            x = self.dropout(x)

        if self.scale_depth > 0:
            hidden_states = hidden_states + x * (
                self.scale_depth / math.sqrt(self.num_layers)
            )  # https://arxiv.org/pdf/2310.02244.pdf
        else:
            hidden_states = hidden_states + x

        return hidden_states


class MoEFeedForward(bmt.DistributedModule):
    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        activate_fn: str = "silu",
        dtype=torch.bfloat16,
        dropout_p: Optional[float] = None,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        num_experts: int = 8,
        top_k: int = 2,
        top_p: float = 0.3,
        routing_strategy: str = "topk",
        num_shared_experts: int = 0,
        ffn_gated: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        if num_shared_experts > 0 and routing_strategy == "topp":
            raise NotImplementedError()

        self.num_shared_experts = num_shared_experts
        if self.num_shared_experts > 0:
            self.shared_experts = FeedForward(
                dim_model,
                dim_ff * self.num_shared_experts,
                activate_fn=activate_fn,
                dtype=dtype,
                dropout_p=dropout_p,
                tp=tp,
                scale=scale,
                init_std=init_std,
                scale_width=scale_width,
                ffn_gated=ffn_gated,
                eps=eps,
            )

        self.num_routed_experts = num_experts - num_shared_experts
        assert top_k > num_shared_experts
        self.top_k_routed = top_k - num_shared_experts
        self.experts = FeedForward(
            dim_model,
            dim_ff,
            activate_fn=activate_fn,
            dtype=dtype,
            dropout_p=dropout_p,
            tp=tp,
            scale=scale,
            init_std=init_std,
            scale_width=scale_width,
            ffn_gated=ffn_gated,
            num_experts=self.num_routed_experts,
            eps=eps,
        )
        if routing_strategy in ["topk", "topp", "sparse_mixer", "relu", "relu_rms"]:
            router_cls = {
                "topk": TopKRouter,
                "topp": TopPRouter,
                "sparse_mixer": SparseMixerV2,
                "relu": ReLURouter,
                "relu_rms": ReLURouter,
            }[routing_strategy]
            if routing_strategy == "topp":
                self.router = router_cls(
                    dim_model=dim_model,
                    num_experts=self.num_routed_experts,
                    top_p=top_p,
                    dtype=dtype,
                )
            elif routing_strategy == "relu_rms":
                self.router = router_cls(
                    dim_model=dim_model,
                    num_experts=self.num_routed_experts,
                    top_k=self.top_k_routed,
                    dtype=dtype,
                    rms_norm=True,
                    eps=eps,
                )
            else:
                self.router = router_cls(
                    dim_model=dim_model,
                    num_experts=self.num_routed_experts,
                    top_k=self.top_k_routed,
                    dtype=dtype,
                )
            self.mix_calculator = FastTopKCalculator(
                num_experts=self.num_routed_experts,
            )
        elif routing_strategy == "expert_choice":
            self.moe_gate = TopKExpertChoiceRouter(
                dim_model=dim_model,
                num_experts=self.num_routed_experts,
                dtype=dtype,
                top_k=self.top_k_routed,
            )
            self.mix_calculator = ExpertChoiceCalculator(
                num_experts=self.num_routed_experts,
            )
        else:
            raise NotImplementedError(f"strategy {routing_strategy} is not implemented!!!")

    def forward(self, hidden_states, seq_mask: torch.Tensor = None):
        gate_output = self.router.forward(hidden_states)
        ActivationContext.stat_act_rate(gate_output["topk_indices"], self.num_shared_experts, self.num_routed_experts, seq_mask)
        y = self.mix_calculator.forward(
            hidden_states=hidden_states,
            topk_indices=gate_output["topk_indices"].contiguous(),
            topk_weights=gate_output["topk_scores"],
            experts=self.experts
        )
        if self.num_shared_experts > 0:
            y = y + self.shared_experts(hidden_states)
        return {
            "hidden_states": y,
            "balance_loss": gate_output["balance_loss"],
            "load": gate_output["load"],
            "router_entropy": gate_output["router_entropy"],
        }


class FFNBlock(torch.nn.Module):
    """The whole feed-forward block. A sequence of operation. Consists of layernorm, feed-forward and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        dtype (optional): Defaults to torch.bfloat16.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        activate_fn: str,
        dtype=torch.bfloat16,
        eps: float = 1e-6,
        dropout_p: Optional[float] = 0,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        scale_depth: float = -1,
        layer_id: int = 0,
        num_layers: int = 0,
        num_experts: int = 8,
        ffn_gated: bool = True,
        top_k: int = 2,
        top_p: float = 0.3,
        routing_strategy: str = "topk",
        num_shared_experts: int = 0,
    ):
        super().__init__()

        self.layernorm_before_ffn = LayerNorm(
            dim_model,
            dtype=dtype,
            eps=eps,
        )

        if num_experts == 0:
            self.ffn = FeedForward(
                dim_model,
                dim_ff,
                activate_fn=activate_fn,
                dtype=dtype,
                dropout_p=dropout_p,
                tp=tp,
                scale=scale,
                init_std=init_std,
                scale_width=scale_width,
                ffn_gated=ffn_gated,
                eps=eps,
            )
        else:
            self.ffn = MoEFeedForward(
                dim_model=dim_model,
                dim_ff=dim_ff,
                activate_fn=activate_fn,
                dtype=dtype,
                dropout_p=dropout_p,
                tp=tp,
                scale=scale,
                init_std=init_std,
                scale_width=scale_width,
                num_experts=num_experts,
                top_k=top_k,
                top_p=top_p,
                routing_strategy=routing_strategy,
                num_shared_experts=num_shared_experts,
                ffn_gated=ffn_gated,
                eps=eps,
            )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.scale = scale
        self.scale_depth = scale_depth
        self.num_layers = num_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_mask: torch.Tensor,
    ):
        x = self.layernorm_before_ffn(hidden_states)
        ffn_output = self.ffn(x, seq_mask)
        if isinstance(ffn_output, dict):
            x, balance_loss, load, router_entropy = ffn_output["hidden_states"], ffn_output["balance_loss"], ffn_output["load"], ffn_output["router_entropy"]
        else:
            x, balance_loss, load, router_entropy = ffn_output, None, None, None
        if self.dropout is not None:
            x = self.dropout(x)

        if self.scale_depth > 0:
            hidden_states = hidden_states + x.view_as(hidden_states) * (
                self.scale_depth / math.sqrt(self.num_layers)
            )  # https://arxiv.org/pdf/2310.02244.pdf
        else:
            hidden_states = hidden_states + x.view_as(hidden_states)

        return {
            "hidden_states": hidden_states,
            "balance_loss": balance_loss,
            "load": load,
            "router_entropy": router_entropy,
        }


class TransformerBlock(torch.nn.Module):
    """The whole transformer block. A sequence of operation. Consists of self-attention block[, cross-attention block] and feed-forward block.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.bfloat16.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        num_heads: int,
        num_kv_heads: int,
        dim_head: int,
        activate_fn: str = "silu",
        dtype=torch.bfloat16,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        scale_depth: float = -1,
        qk_norm: bool = False,
        layer_id: int = 0,
        num_layers: int = 0,
        num_experts: int = 8,
        top_k: int = 2,
        top_p: float = 0.3,
        routing_strategy: str = "topk",
        num_shared_experts: int = 0,
        ffn_gated: bool = True,
        attention_type: str = "vanilla",
        dim_q_lora: int = 0,
        dim_kv_lora: int = 0,
        dim_q_nope_head: int = 0,
        dim_q_pe_head: int = 0,
        dim_v_head: int = 0,
    ):
        super().__init__()

        self.self_att = SelfAttentionBlock(
            dim_model=dim_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dim_head=dim_head,
            dtype=dtype,
            eps=eps,
            dropout_p=dropout_p,
            tp=tp,
            scale=scale,
            init_std=init_std,
            scale_width=scale_width,
            scale_depth=scale_depth,
            qk_norm=qk_norm,
            layer_id=layer_id,
            num_layers=num_layers,
            attention_type=attention_type,
            dim_q_lora=dim_q_lora,
            dim_kv_lora=dim_kv_lora,
            dim_q_nope_head=dim_q_nope_head,
            dim_q_pe_head=dim_q_pe_head,
            dim_v_head=dim_v_head,
        )

        self.ffn = FFNBlock(
            dim_model=dim_model,
            dim_ff=dim_ff,
            activate_fn=activate_fn,
            dtype=dtype,
            eps=eps,
            dropout_p=dropout_p,
            tp=tp,
            scale=scale,
            init_std=init_std,
            scale_width=scale_width,
            scale_depth=scale_depth,
            layer_id=layer_id,
            num_layers=num_layers,
            num_experts=num_experts,
            ffn_gated=ffn_gated,
            top_k=top_k,
            top_p=top_p,
            routing_strategy=routing_strategy,
            num_shared_experts=num_shared_experts,
        )

    def forward(
        self,
        self_hidden_states: torch.Tensor,
        self_position_bias: Optional[torch.Tensor] = None,  # TODO
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ):
        # (batch, dim_model, seq_self)
        hidden_states = self.self_att(
            self_hidden_states,
            position_bias=self_position_bias,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids,
        )

        # (batch, dim_model, seq_self)
        ffn_output = self.ffn(hidden_states, seq_mask)

        return ffn_output["hidden_states"], ffn_output["balance_loss"], ffn_output["load"], ffn_output["router_entropy"]


class Encoder(bmt.DistributedModule):
    """Layers of encoder transformer blocks plus an final layernorm.

    Args:
        num_layers (int): number of layers.
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.bfloat16.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        num_layers: int,
        dim_model: int,
        dim_ff: int,
        num_heads: int,
        dim_head: int,
        num_kv_heads: int = -1,
        activate_fn: str = "silu",
        dtype: torch.dtype = torch.bfloat16,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        scale_depth: float = -1,
        qk_norm: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        top_p: float = 0.3,
        routing_strategy: str = "topk",
        num_shared_experts: int = 0,
        disable_checkpoint=None,
        ffn_gated: bool = True,
        attention_type: str = "vanilla",
        dim_q_lora: int = 0,
        dim_kv_lora: int = 0,
        dim_q_nope_head: int = 0,
        dim_q_pe_head: int = 0,
        dim_v_head: int = 0,
    ):
        super().__init__()
        if num_kv_heads == -1:
            num_kv_heads = num_heads
        self.num_layers = num_layers

        self.layers = bmt.TransformerBlockList(
            [
                bmt.CheckpointBlock(
                    TransformerBlock(
                        dim_model=dim_model,
                        dim_ff=dim_ff,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        dim_head=dim_head,
                        activate_fn=activate_fn,
                        dtype=dtype,
                        eps=eps,
                        dropout_p=dropout_p,
                        tp=tp,
                        scale=scale,
                        init_std=init_std,
                        scale_width=scale_width,
                        scale_depth=scale_depth,
                        qk_norm=qk_norm,
                        layer_id=layer_id,
                        num_layers=num_layers,
                        num_experts=num_experts,
                        top_k=top_k,
                        top_p=top_p,
                        routing_strategy=routing_strategy,
                        num_shared_experts=num_shared_experts,
                        ffn_gated=ffn_gated,
                        attention_type=attention_type,
                        dim_q_lora=dim_q_lora,
                        dim_kv_lora=dim_kv_lora,
                        dim_q_nope_head=dim_q_nope_head,
                        dim_q_pe_head=dim_q_pe_head,
                        dim_v_head=dim_v_head,
                    ),
                    use_checkpoint=disable_checkpoint is None or layer_id not in disable_checkpoint,
                )
                for layer_id in range(num_layers)
            ]
        )
        self.output_layernorm = LayerNorm(dim_norm=dim_model, dtype=dtype, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: torch.Tensor = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ):
        moe_info = {"balance_loss": [], "load": [], "router_entropy": []}
        for idx in range(self.num_layers):
            hidden_states, moe_loss, moe_load, router_entropy = self.layers[idx](
                hidden_states,
                position_bias,
                cu_seqlens,
                max_seqlen,
                position_ids,
                seq_mask,
            )
            if moe_loss is not None:
                moe_info["balance_loss"].append(moe_loss)
                moe_info["load"].append(moe_load)
                moe_info["router_entropy"].append(router_entropy)
        hidden_states = self.output_layernorm(hidden_states)
        return hidden_states, moe_info


class Dragonfly(bmt.DistributedModule):
    def __init__(self, config: DragonflyConfig):
        super().__init__()

        self.encoder = Encoder(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            dim_ff=config.dim_ff,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dim_head=config.dim_head,
            activate_fn=config.activate_fn,
            dtype=config.dtype,
            eps=config.eps,
            dropout_p=config.dropout_p,
            tp=config.tp,
            scale=config.scale,
            init_std=config.init_std,
            scale_width=config.scale_width,
            scale_depth=config.scale_depth,
            qk_norm=config.qk_norm,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            top_p=config.moe_top_p,
            routing_strategy=config.moe_routing_strategy,
            num_shared_experts=config.num_shared_experts,
            disable_checkpoint=config.disable_checkpoint,
            ffn_gated=config.ffn_gated,
            attention_type=config.attention_type,
            dim_q_lora=config.dim_q_lora,
            dim_kv_lora=config.dim_kv_lora,
            dim_q_nope_head=config.dim_q_nope_head,
            dim_q_pe_head=config.dim_q_pe_head,
            dim_v_head=config.dim_v_head,
        )

        self.input_embedding = Embedding(
            vocab_size=config.vocab_size,
            embedding_size=config.dim_model,
            dtype=config.dtype,
            init_std=config.init_std,
            tp=config.tp,
            scale=config.scale,
            scale_emb=config.scale_emb,
            scale_width=config.scale_width,
        )

        if config.attention_type == "mla":
            rope_dim = config.dim_q_pe_head
        elif config.attention_type == "vanilla":
            rope_dim = config.dim_head
        else:
            raise NotImplementedError(f"invalid attention_type {config.attention_type}")
        self.position_bias = ScaledRotaryEmbeddingESM(
            dim=rope_dim,
            max_position_embeddings=config.max_length,
            base=config.base,
            pose_prob=config.pose_prob,
            pose_scaling_factor=config.pose_scaling_factor,
            scaling_type=config.rope_scaling_type,
            rope_scaling_factor=config.rope_scaling_factor,
            original_max_position_embeddings=config.orig_max_length,
            dynamic_scaling_seq_len=config.max_length,  # disable dynamic scaling
            persistent=False,
            device="cuda",
        )

        if config.tie_lm_head is False:
            self.lm_head = Embedding(
                vocab_size=config.vocab_size,
                embedding_size=config.dim_model,
                dtype=config.dtype,
                init_std=config.init_std,
                scale=config.scale,
                scale_width=config.scale_width,
                tp=config.tp,
            )

        self.config = config

    def forward(
        self,
        input: torch.Tensor,  # (batch, seqlen) int32
        cu_seqlens: torch.Tensor = None,  # (real_batch+2) int32
        max_seqlen: int = None,
        position_ids: torch.Tensor = None,  # (batch, seqlen) int32
        seq_mask: torch.Tensor = None,
    ):
        hidden_states = self.input_embedding(input)

        assert cu_seqlens is not None, "cu_seqlens are needed in Flash Attention cuda impl"
        hidden_states, moe_info = self.encoder(
            hidden_states,
            position_bias=self.position_bias,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids,
            seq_mask=seq_mask,
        )

        if self.config.tie_lm_head is True:
            logits = self.input_embedding.projection(hidden_states)
        else:
            logits = self.lm_head.projection(hidden_states)

        return logits, moe_info
