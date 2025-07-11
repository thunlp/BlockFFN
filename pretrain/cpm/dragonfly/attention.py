import math
import torch
import bmtrain as bmt
from typing import Optional, Union, Callable
from flash_attn.flash_attn_interface import flash_attn_varlen_func

from .layer_norm import LayerNorm
from .linear import Linear


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
        layer_id: int = 0,
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
