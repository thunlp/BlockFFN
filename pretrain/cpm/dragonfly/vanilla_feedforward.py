import math
import torch
import bmtrain as bmt
from .linear import Linear
from typing import Optional
from .activation_context import ActivationContext
from .activation_recorder import ActivationRecorder, ActivationCollector
from .activation_function import get_activation_fn


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

        self.act = get_activation_fn(activate_fn)

    def forward(self, x: torch.Tensor):
        """This model inherits from bmt.DistributedModule.
            Transform an input tensor from one feature space to another via a nonlinear operation

        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): Tensor that will be subject to nonlinear operations.

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_ff)``)

        """  # noqa: E501
        inputs = x
        gate_score = self.act(self.w_0(x))
        ActivationContext.append_intermediate(gate_score)
        ActivationContext.stat_window_activation(gate_score, valid_loss_mask=None)
        x = self.w_1(x)

        x = gate_score * x
        ActivationCollector.record_activations(inputs, x, gate_score)
        return x


class VanillaFeedForward(bmt.DistributedModule):
    r"""Vanilla FeedForward module"""  # noqa: E501
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
        layer_id: int = 0,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.w_in = DenseGatedACT(
            dim_in=dim_model,
            dim_ff=dim_ff,
            activate_fn=activate_fn,
            dtype=dtype,
            scale=scale,
            init_std=init_std,
            scale_width=scale_width,
            num_experts=num_experts,
        )

        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        _std = init_std / math.sqrt(scale_width) if scale else init_std
        self.w_out = Linear(dim_in=dim_ff, dim_out=dim_model, dtype=dtype, init_std=_std, num_experts=num_experts)
        self.layer_id = layer_id
        self.tp = tp

    def forward(self, x: torch.Tensor, **kwargs):
        ActivationContext.append_input(x)
        x = self.w_in(x)
        l1_loss = ActivationContext.l1_regularization(x, self.layer_id)

        if self.dropout is not None:
            x = self.dropout(x)

        ActivationRecorder.record_fp_mlp_label(self.layer_id, x)

        x = self.w_out(x)
        ActivationContext.append_output(x)
        balance_loss = torch.tensor(0.).to(x)
        router_entropy = torch.tensor(0.).to(x)
        transfer_loss = torch.tensor(0.).to(x)
        chunk_loss = torch.tensor(0.).to(x)
        token_balance_loss = torch.tensor(0.).to(x)
        return x, l1_loss, balance_loss, router_entropy, transfer_loss, chunk_loss, token_balance_loss
