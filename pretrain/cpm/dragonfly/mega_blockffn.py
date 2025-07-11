import torch
import bmtrain as bmt
from typing import Optional
from .activation_context import ActivationContext
from .activation_recorder import ActivationRecorder

from megablocks import dMoE
from bmtrain.distributed.ops import all_gather, reduce_scatter
from .mega_moe_modules import MegaMoELinearActiveGate, BlockSparseMLP


class MegaBLockFeedForward(dMoE):
    def __init__(
            self,
            dim_model: int,
            dim_ff: int,
            num_experts: int,
            router_activate_fn: str = "relu",
            ffn_activate_fn: str = "silu",
            dtype=torch.float32,
            dropout_p: Optional[float] = None,
            tp: int = 0,
            scale: bool = False,
            init_std: float = 0.02,
            scale_width: float = 1.0,
            layer_id: int = 0,
    ):
        super().__init__(
            dim_model=dim_model,
            dim_ff=dim_ff,
            num_experts=num_experts // bmt.config["tp_size"],
            router=MegaMoELinearActiveGate(
                dim_model=dim_model, num_expert=num_experts // bmt.config["tp_size"],
                activate_fn=router_activate_fn, dtype=dtype, scale=scale, init_std=init_std,
                scale_width=scale_width, tp=tp,
            ),
            mlp=BlockSparseMLP(
                dim_model=dim_model, dim_ff=dim_ff, num_experts=num_experts // bmt.config["tp_size"],
                activate_fn=ffn_activate_fn, dtype=dtype, scale=scale, init_std=init_std,
                scale_width=scale_width, tp=tp,
            ),
        )

        self.layer_id = layer_id
        self.tp = tp  # not used at present

    def forward(self, x: torch.Tensor, **kwargs):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of feed-forward module.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of feed-forward module.
        """  # noqa: E501
        shape = x.shape
        if self.tp:
            x = all_gather(x, bmt.config["tp_comm"])
        ActivationContext.append_input(x)
        x = x.flatten(0, 1)
        router_score = self.router(x)
        router_score = router_score / (torch.sum(router_score, dim=-1, keepdim=True) + 1e-5)

        # add router entropy
        router_entropy = torch.sum(-router_score * torch.log(router_score + 1e-5), dim=-1)
        router_entropy = router_entropy.mean()

        # add load balance
        _, num_experts = router_score.shape
        act_flag = torch.gt(router_score, 0)
        load = act_flag.sum(dim=0)
        load_mean = load / (torch.sum(act_flag) + 1e-5)
        importance_mean = router_score.mean(dim=0)
        balance_loss = num_experts * torch.sum(importance_mean * load_mean)

        ActivationContext.append_intermediate(router_score)
        l1_loss = ActivationContext.l1_regularization(router_score, self.layer_id)
        ActivationRecorder.record_fp_mlp_label(self.layer_id, router_score)

        x, tokens_per_expert = self.block_sparse_forward(x, router_score)
        if self.tp:
            x = reduce_scatter(x, "sum", bmt.config["tp_comm"])
        x = x.view(shape)
        ActivationContext.append_output(x)

        return x, l1_loss, balance_loss, router_entropy
