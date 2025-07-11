import torch
import bmtrain as bmt
from typing import Optional
from .vanilla_feedforward import VanillaFeedForward
from .moe_modules import TopkRouter, FastTopkCalculator


class MoEFeedForward(bmt.DistributedModule):
    def __init__(
        self,
        dim_model: int,
        dim_expert: int,
        num_experts: int,
        ffn_activate_fn: str = "silu",
        dtype=torch.bfloat16,
        dropout_p: Optional[float] = None,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        top_k: int = 2,
        layer_id: int = 0,
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.experts = VanillaFeedForward(
            dim_model,
            dim_expert,
            activate_fn=ffn_activate_fn,
            dtype=dtype,
            dropout_p=dropout_p,
            tp=tp,
            scale=scale,
            init_std=init_std,
            scale_width=scale_width,
            num_experts=num_experts,
        )
        self.router = TopkRouter(
            dim_model=dim_model,
            num_experts=num_experts,
            top_k=top_k,
            dtype=dtype,
        )
        self.mix_calculator = FastTopkCalculator(
            num_experts=num_experts,
            top_k=top_k
        )
        self.layer_id = layer_id

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        gate_output = self.router.forward(hidden_states)
        y = self.mix_calculator.forward(
            hidden_states=hidden_states,
            topk_indices=gate_output["topk_indices"].contiguous(),
            topk_weights=gate_output["topk_scores"],
            experts=self.experts
        )
        return y
