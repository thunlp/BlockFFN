import math
import torch
import bmtrain as bmt
from typing import Tuple
import torch.nn.functional as F


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
