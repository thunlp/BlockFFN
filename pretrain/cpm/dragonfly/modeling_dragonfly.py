# typing: strict
# coding=utf-8
# Copyright 2023 ModelBest Inc.

import math
import torch
import bmtrain as bmt
from typing import Optional

from .configuration_dragonfly import DragonflyConfig
from .embedding import ScaledRotaryEmbeddingESM, Embedding
from .layer_norm import LayerNorm
from .moe_modules import MoELinearActiveGate
from .attention import Attention, MultiLatentAttention
from .activation_recorder import ActivationRecorder
from .vanilla_feedforward import VanillaFeedForward
from .blockffn import BLockFeedForward
from .moe_feedforward import MoEFeedForward
from .mega_blockffn import MegaBLockFeedForward


class SelfAttentionBlock(bmt.DistributedModule):
    """The whole cross-attention block. A sequence of operation. Consists of layernorm, self-attention and residual connection"""  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
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
        config: DragonflyConfig = None,
    ):
        super().__init__()

        self.layernorm_before_attention = LayerNorm(
            dim_model,
            dtype=dtype,
            eps=eps,
        )

        if config.attention_type == "vanilla":
            self.self_attention = Attention(
                dim_model=dim_model,
                num_heads=num_heads,
                num_kv_heads=config.num_kv_heads,
                dim_head=config.dim_head,
                dtype=dtype,
                dropout_p=dropout_p,
                tp=tp,
                scale=scale,
                init_std=init_std,
                scale_width=scale_width,
                qk_norm=qk_norm,
                layer_id=layer_id,
            )
        elif config.attention_type == "mla":
            self.self_attention = MultiLatentAttention(
                dim_model=dim_model,
                num_heads=num_heads,
                dim_q_nope_head=config.dim_q_nope_head,
                dim_q_pe_head=config.dim_q_pe_head,
                dim_v_head=config.dim_v_head,
                dim_q_lora=config.dim_q_lora,
                dim_kv_lora=config.dim_kv_lora,
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
            raise NotImplementedError(f"invalid attention_type {config.attention_type}")

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.scale = scale
        self.scale_depth = scale_depth
        self.num_layers = num_layers
        self.layer_id = layer_id

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: ScaledRotaryEmbeddingESM,
        cu_seqlens: torch.Tensor,
        max_seqlen: int = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        ActivationRecorder.record_fp_attn_query(self.layer_id, hidden_states)

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


class FFNBlock(torch.nn.Module):
    """The whole feed-forward block. A sequence of operation. Consists of layernorm, feed-forward and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        dim_expert (int): the intermediate dimension per expert.
        num_expert (int): the number of intermediate expert pairs (up + down).
        dtype (optional): Defaults to torch.bfloat16.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        dim_expert: int,
        num_expert: int,
        router_activate_fn: str = "relu",
        ffn_activate_fn: str = "silu",
        expert_gated: bool = False,
        ffn_type: str = "block",
        norm_after_router: str = "sum",
        norm_scale: float = 1.0,
        dtype=torch.bfloat16,
        eps: float = 1e-6,
        dropout_p: Optional[float] = 0,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        scale_depth: float = -1,
        top_k: int = 2,
        layer_id: int = 0,
        num_layers: int = 0,
    ):
        super().__init__()

        self.layernorm_before_ffn = LayerNorm(
            dim_model,
            dtype=dtype,
            eps=eps,
        )

        if "block" in ffn_type and ffn_type != "megablock":
            self.ffn = BLockFeedForward(
                dim_model,
                dim_expert=dim_expert,
                num_expert=num_expert,
                router_activate_fn=router_activate_fn,
                ffn_activate_fn=ffn_activate_fn,
                expert_gated=expert_gated,
                dtype=dtype,
                dropout_p=dropout_p,
                tp=tp,
                scale=scale,
                init_std=init_std,
                scale_width=scale_width,
                adaptive=(ffn_type == "adpblock"),
                use_residual=(ffn_type == "resblock"),
                use_head=(ffn_type == "headblock"),
                use_linear=("block_" in ffn_type),
                gate_policy=(ffn_type.split("_")[-1] if "block_" in ffn_type and ffn_type != "block_linear" else "null"),
                norm_after_router=norm_after_router,
                norm_scale=norm_scale,
                eps=eps,
                layer_id=layer_id,
            )
        elif ffn_type == "vanilla":
            self.ffn = VanillaFeedForward(
                dim_model,
                dim_ff,
                activate_fn=ffn_activate_fn,
                dtype=dtype,
                dropout_p=dropout_p,
                tp=tp,
                scale=scale,
                init_std=init_std,
                scale_width=scale_width,
                layer_id=layer_id,
            )
        elif ffn_type == "megablock":
            self.ffn = MegaBLockFeedForward(
                dim_model,
                dim_ff=dim_expert,
                num_experts=num_expert,
                router_activate_fn=router_activate_fn,
                ffn_activate_fn=ffn_activate_fn,
                dtype=dtype,
                dropout_p=dropout_p,
                tp=tp,
                scale=scale,
                init_std=init_std,
                scale_width=scale_width,
                layer_id=layer_id,
            )
        elif ffn_type == "moe":
            self.ffn = MoEFeedForward(
                dim_model,
                dim_expert=dim_expert,
                num_experts=num_expert,
                ffn_activate_fn=ffn_activate_fn,
                dtype=dtype,
                dropout_p=dropout_p,
                tp=tp,
                scale=scale,
                init_std=init_std,
                scale_width=scale_width,
                top_k=top_k,
                layer_id=layer_id,
            )
        else:
            raise NotImplementedError(f"invalid ffn_type: {ffn_type}")

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.scale = scale
        self.scale_depth = scale_depth
        self.num_layers = num_layers
        self.layer_id = layer_id

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor = None,
        sub_router_scores: torch.Tensor = None,
        segment_shift_indices: torch.Tensor = None,
        seq_mask: torch.Tensor = None,
    ):
        """
        Args:
            hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``): Hidden states before feed forward layer.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of feed-forward block

        """  # noqa: E501
        ActivationRecorder.record_fp_mlp_query(self.layer_id, hidden_states)

        x = self.layernorm_before_ffn(hidden_states)
        if residual is not None:
            residual = self.layernorm_before_ffn(residual)
        x, l1_loss, balance_loss, router_entropy, transfer_loss, chunk_loss, token_balance_loss = self.ffn(
            x, residual=residual, sub_router_scores=sub_router_scores,
            segment_shift_indices=segment_shift_indices, seq_mask=seq_mask,
        )
        if self.dropout is not None:
            x = self.dropout(x)

        if self.scale_depth > 0:
            hidden_states = hidden_states + x.view_as(hidden_states) * (
                self.scale_depth / math.sqrt(self.num_layers)
            )  # https://arxiv.org/pdf/2310.02244.pdf
        else:
            hidden_states = hidden_states + x.view_as(hidden_states)

        return hidden_states, l1_loss, balance_loss, router_entropy, transfer_loss, chunk_loss, token_balance_loss


class TransformerBlock(torch.nn.Module):
    """The whole transformer block. A sequence of operation. Consists of self-attention block[, cross-attention block] and feed-forward block"""  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        dim_expert: int,
        num_expert: int,
        num_heads: int,
        router_activate_fn: str = "relu",
        ffn_activate_fn: str = "silu",
        expert_gated: bool = False,
        ffn_type: str = "block",
        norm_after_router: str = "sum",
        norm_scale: float = 1.0,
        dtype=torch.bfloat16,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        scale_depth: float = -1,
        qk_norm: bool = False,
        top_k: int = 2,
        layer_id: int = 0,
        num_layers: int = 0,
        config: DragonflyConfig = None,
    ):
        super().__init__()

        self.self_att = SelfAttentionBlock(
            dim_model=dim_model,
            num_heads=num_heads,
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
            config=config,
        )

        self.ffn = FFNBlock(
            dim_model=dim_model,
            dim_ff=dim_ff,
            dim_expert=dim_expert,
            num_expert=num_expert,
            router_activate_fn=router_activate_fn,
            ffn_activate_fn=ffn_activate_fn,
            expert_gated=expert_gated,
            ffn_type=ffn_type,
            norm_after_router=norm_after_router,
            norm_scale=norm_scale,
            dtype=dtype,
            eps=eps,
            dropout_p=dropout_p,
            tp=tp,
            scale=scale,
            init_std=init_std,
            scale_width=scale_width,
            scale_depth=scale_depth,
            top_k=top_k,
            layer_id=layer_id,
            num_layers=num_layers,
        )

        self.layer_id = layer_id
        ActivationRecorder.add_recorder(dim_model, num_expert, num_heads, layer_id)

    def forward(
        self,
        self_hidden_states: torch.Tensor,
        self_position_bias: Optional[torch.Tensor] = None,  # TODO
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        sub_router_scores: Optional[torch.Tensor] = None,
        segment_shift_indices: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ):
        # (batch, dim_model, seq_self)
        residual = self_hidden_states
        hidden_states = self.self_att(
            self_hidden_states,
            position_bias=self_position_bias,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids,
        )

        # (batch, dim_model, seq_self)
        hidden_states, l1_loss, balance_loss, router_entropy, transfer_loss, chunk_loss, token_balance_loss = self.ffn(
            hidden_states, residual, sub_router_scores, segment_shift_indices, seq_mask,
        )

        ActivationRecorder.check_id_ok(self.layer_id)

        return hidden_states, l1_loss, balance_loss, router_entropy, transfer_loss, chunk_loss, token_balance_loss


class Encoder(bmt.DistributedModule):
    """Layers of encoder transformer blocks plus an final layernorm"""  # noqa: E501
    def __init__(
        self,
        num_layers: int,
        dim_model: int,
        dim_ff: int,
        dim_expert: int,
        num_expert: int,
        num_heads: int,
        router_activate_fn: str = "relu",
        ffn_activate_fn: str = "silu",
        expert_gated: bool = False,
        ffn_type: str = "block",
        norm_after_router: str = "sum",
        norm_scale: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        scale_depth: float = -1,
        qk_norm: bool = False,
        top_k: int = 2,
        use_checkpoint: bool = True,
        config: DragonflyConfig = None,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.layers = bmt.TransformerBlockList(
            [
                bmt.CheckpointBlock(
                    TransformerBlock(
                        dim_model=dim_model,
                        dim_ff=dim_ff,
                        dim_expert=dim_expert,
                        num_expert=num_expert,
                        num_heads=num_heads,
                        router_activate_fn=router_activate_fn,
                        ffn_activate_fn=ffn_activate_fn,
                        expert_gated=expert_gated,
                        ffn_type=ffn_type,
                        norm_after_router=norm_after_router,
                        norm_scale=norm_scale,
                        dtype=dtype,
                        eps=eps,
                        dropout_p=dropout_p,
                        tp=tp,
                        scale=scale,
                        init_std=init_std,
                        scale_width=scale_width,
                        scale_depth=scale_depth,
                        qk_norm=qk_norm,
                        top_k=top_k,
                        layer_id=layer_id,
                        num_layers=num_layers,
                        config=config,
                    ),
                    use_checkpoint=use_checkpoint
                )
                for layer_id in range(num_layers)
            ]
        )
        self.output_layernorm = LayerNorm(dim_norm=dim_model, dtype=dtype, eps=eps)

        self.num_expert = num_expert
        self.dim_model = dim_model
        self.use_head_block = config.ffn_type == "headblock"
        if self.use_head_block:
            _std = init_std / math.sqrt(scale_width) if scale else init_std
            self.layernorm_before_router = LayerNorm(
                dim_model,
                dtype=dtype,
                eps=eps,
            )
            self.all_moe_router = MoELinearActiveGate(
                dim_model=dim_model,
                num_expert=num_expert * num_layers,
                activate_fn=router_activate_fn,
                dtype=dtype,
                init_std=_std,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: torch.Tensor = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        segment_shift_indices: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ):
        total_l1_loss = torch.tensor(0.).to(hidden_states)
        ave_balance_loss = torch.tensor(0.).to(hidden_states)
        ave_router_entropy = torch.tensor(0.).to(hidden_states)
        ave_transfer_loss = torch.tensor(0.).to(hidden_states)
        ave_chunk_loss = torch.tensor(0.).to(hidden_states)
        ave_token_balance_loss = torch.tensor(0.).to(hidden_states)

        raw_all_router_scores, all_router_scores = None, None
        if self.use_head_block:
            norm_hidden = self.layernorm_before_router(hidden_states.view(-1, self.dim_model))
            raw_all_router_scores, all_router_scores = self.all_moe_router(norm_hidden)

        for idx, layer in enumerate(self.layers):
            sub_router_scores = None
            if self.use_head_block:
                sub_router_scores = raw_all_router_scores[:, self.num_expert*idx:self.num_expert*(idx+1)], \
                    all_router_scores[:, self.num_expert*idx:self.num_expert*(idx+1)]
            hidden_states, l1_loss, balance_loss, router_entropy, transfer_loss, chunk_loss, token_balance_loss = layer(
                hidden_states,
                position_bias,
                cu_seqlens,
                max_seqlen,
                position_ids,
                sub_router_scores,
                segment_shift_indices,
                seq_mask,
            )
            total_l1_loss += l1_loss
            ave_balance_loss += balance_loss
            ave_router_entropy += router_entropy
            ave_transfer_loss += transfer_loss
            ave_chunk_loss += chunk_loss
            ave_token_balance_loss += token_balance_loss
        ave_balance_loss /= len(self.layers)
        ave_router_entropy /= len(self.layers)
        ave_transfer_loss /= len(self.layers)
        ave_chunk_loss /= len(self.layers)
        ave_token_balance_loss /= len(self.layers)
        hidden_states = self.output_layernorm(hidden_states)
        return hidden_states, total_l1_loss, ave_balance_loss, ave_router_entropy, ave_transfer_loss, ave_chunk_loss, ave_token_balance_loss


class Dragonfly(bmt.DistributedModule):
    def __init__(self, config: DragonflyConfig):
        super().__init__()

        self.encoder = Encoder(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            dim_ff=config.dim_ff,
            dim_expert=config.dim_expert,
            num_expert=config.num_expert,
            num_heads=config.num_heads,
            router_activate_fn=config.router_activate_fn,
            ffn_activate_fn=config.ffn_activate_fn,
            expert_gated=config.expert_gated,
            ffn_type=config.ffn_type,
            norm_after_router=config.norm_after_router,
            norm_scale=config.norm_scale,
            dtype=config.dtype,
            eps=config.eps,
            dropout_p=config.dropout_p,
            tp=config.tp,
            scale=config.scale,
            init_std=config.init_std,
            scale_width=config.scale_width,
            scale_depth=config.scale_depth,
            qk_norm=config.qk_norm,
            top_k=config.moe_top_k,
            use_checkpoint=config.use_checkpoint,
            config=config,
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
        segment_shift_indices: torch.Tensor = None,
        seq_mask: torch.Tensor = None,
    ):
        hidden_states = self.input_embedding(input)

        assert cu_seqlens is not None, "cu_seqlens are needed in Flash Attention cuda impl"
        hidden_states, l1_loss, balance_loss, router_entropy, transfer_loss, chunk_loss, token_balance_loss = self.encoder(
            hidden_states,
            position_bias=self.position_bias,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids,
            segment_shift_indices=segment_shift_indices,
            seq_mask=seq_mask,
        )

        if self.config.tie_lm_head is True:
            logits = self.input_embedding.projection(hidden_states)
        else:
            logits = self.lm_head.projection(hidden_states)

        return logits, l1_loss, balance_loss, router_entropy, transfer_loss, chunk_loss, token_balance_loss
