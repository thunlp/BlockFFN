import os
import math
import torch
import bmtrain as bmt
from .linear import Linear
from typing import Optional
from .activation_context import ActivationContext
from .activation_recorder import ActivationRecorder
from .activation_function import get_activation_fn
from .layer_norm import LayerNorm, rms_layernorm
from .moe_modules import MoELinearActiveGate, MoEUpDownExperts
from cpm.training_utils.value_scheduler import CosineScheduler, LinearScheduler, ExpScheduler


class BLockFeedForward(bmt.DistributedModule):
    r"""FeedForward module"""  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_expert: int,
        num_expert: int,
        router_activate_fn: str = "relu",
        ffn_activate_fn: str = "silu",
        expert_gated: bool = False,
        dtype=torch.bfloat16,
        dropout_p: Optional[float] = None,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        adaptive: bool = False,
        use_residual: bool = False,
        use_head: bool = False,
        use_linear: bool = False,
        gate_policy: str= "null",
        norm_after_router: str = "sum",
        norm_scale: float = 1.0,
        eps: float = 1e-6,
        layer_id: int = 0,
    ):
        super().__init__()

        _std = init_std / math.sqrt(scale_width) if scale else init_std
        if not use_head:
            self.moe_router = MoELinearActiveGate(
                dim_model=dim_model,
                num_expert=num_expert,
                activate_fn="softmax" if adaptive else router_activate_fn,
                activate_kwargs={"dim_norm": num_expert, "dtype": dtype, "eps": eps},
                dtype=dtype,
                init_std=_std,
            )
        self.moe_experts = MoEUpDownExperts(
            dim_model=dim_model,
            dim_expert=dim_expert,
            num_expert=num_expert,
            activate_fn=ffn_activate_fn,
            activate_kwargs={"dim_norm": num_expert * dim_expert, "dtype": dtype, "eps": eps},
            expert_gated=expert_gated,
            use_linear=use_linear,
            dropout_p=dropout_p,
            dtype=dtype,
            init_std=_std,
        )
        self.layer_id = layer_id
        self.tp = tp  # not used at present
        self.adaptive = adaptive
        self.use_residual = use_residual
        self.use_head = use_head
        self.norm_after_router = norm_after_router
        self.norm_scale = norm_scale
        self.norm_eps = eps

        if adaptive:
            dim_adp = dim_model // 4
            self.router_act = get_activation_fn(router_activate_fn)
            self.adp_bias_cls = "exp"
            scheduler_cls = {
                "cosine": CosineScheduler,
                "linear": LinearScheduler,
                "exp":    ExpScheduler,
            }[self.adp_bias_cls]
            self.adp_bias_scheduler = scheduler_cls(
                start_val=4.,
                warmup_iter=0,
                end_iter=5000,
                num_iter=0,
                min_ratio=0.01,
            )
            self.adp_mlp_in = Linear(dim_in=dim_model, dim_out=dim_adp, dtype=dtype, init_std=_std)
            self.adp_act = torch.nn.ReLU()
            self.adp_mlp_out = Linear(dim_in=dim_adp, dim_out=1, dtype=dtype, init_std=_std)

        if self.norm_after_router == "rms":
            router_norm_init_var = float(os.environ.get("ROUTER_NORM_INIT_VAR", 1.0))
            router_norm_type = os.environ.get("ROUTER_NORM_TYPE", "rms")
            bmt.print_rank(f"<<<router_norm_init_var: {router_norm_init_var}; router_norm_type: {router_norm_type}>>>")
            self.router_norm = LayerNorm(
                dim_norm=num_expert, dtype=dtype, eps=eps,
                init_var=router_norm_init_var, norm_type=router_norm_type,
            )
        self.num_expert, self.dim_expert = num_expert, dim_expert

        self.gate_policy = gate_policy
        assert self.gate_policy in ["gate1", "gate2", "relugate2", "gate3", "gate4", "drelu", "fixdrelu", "fixgate2", "null"]
        if self.gate_policy in ["gate2", "relugate2", "drelu", "fixdrelu", "fixgate2"]:
            self.router_gate_proj = Linear(dim_in=dim_model, dim_out=num_expert, dtype=dtype, init_std=_std)
        if self.gate_policy in ["gate1", "gate2", "relugate2"]:
            self.router_up_proj = bmt.DistributedParameter(
                torch.empty((num_expert, dim_expert), dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=0.0, std=init_std),
            )
        if self.gate_policy == "gate3":
            self.router_x_proj = Linear(dim_in=dim_model, dim_out=num_expert, dtype=dtype, init_std=_std)
        if self.gate_policy == "gate4":
            self.router_x_down_proj = Linear(dim_in=dim_model, dim_out=dim_expert, dtype=dtype, init_std=_std)
            self.router_x_up_proj = Linear(dim_in=dim_expert, dim_out=num_expert*dim_expert, dtype=dtype, init_std=_std)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor = None,
        sub_router_scores: torch.Tensor = None,
        segment_shift_indices: torch.Tensor = None,
        seq_mask: torch.Tensor = None,
    ):
        batch_size, seq_len, dim_model = x.shape
        if self.use_residual:
            assert residual is not None and residual.shape == x.shape
            residual = residual.view(batch_size * seq_len, dim_model)
        x = x.view(batch_size * seq_len, dim_model)
        router_input = residual if self.use_residual else x
        ActivationContext.append_input(router_input)
        if self.use_head:
            assert sub_router_scores is not None
            raw_router_score, router_score = sub_router_scores
        else:
            raw_router_score, router_score = self.moe_router(router_input)
        if self.adaptive:
            # router_score is produced by softmax
            adp_bias = self.adp_mlp_out(self.adp_act(self.adp_mlp_in(router_input)))
            # ALERT: Is this reasonable?
            adp_bias = torch.sigmoid(adp_bias - self.adp_bias_scheduler.current_val)
            router_score = self.router_act(router_score - adp_bias)

        raw_router_gate, router_gate = None, None
        if self.gate_policy == "fixdrelu":
            raw_router_gate = self.router_gate_proj(router_input)
            router_gate = torch.relu(raw_router_gate)
            router_score = router_score * router_gate

        # add normalization
        if self.norm_after_router == "none":
            router_ori = router_score
            router_score_scaled = router_score / (torch.sum(router_score, dim=-1, keepdim=True) * self.norm_scale + 1e-5)
        elif self.norm_after_router == "sum":
            router_ori = router_score
            router_score = router_score / (torch.sum(router_score, dim=-1, keepdim=True) * self.norm_scale + 1e-5)
            router_score_scaled = router_score * self.norm_scale
        elif self.norm_after_router == "rms":
            router_ori, router_score = self.router_norm(router_score, output_hidden=True)
            router_score_scaled = router_ori / (torch.sum(router_ori, dim=-1, keepdim=True) + 1e-5)
        else:
            raise NotImplementedError(f"invalid norm_after_router: {self.norm_after_router}")

        if self.gate_policy in ["gate2", "relugate2", "drelu"]:
            router_gate = self.router_gate_proj(router_input)
            if self.gate_policy in ["relugate2", "drelu"]:
                router_gate = torch.relu(router_gate)
            router_score = router_score * router_gate
        if self.gate_policy == "gate3":
            router_gate = torch.relu(self.router_x_proj(router_input))
            router_score = router_score * router_gate
        if self.gate_policy == "fixgate2":
            router_gate = self.router_gate_proj(router_input)
            router_score = router_score * router_gate

        # add router entropy
        router_entropy = torch.sum(-router_score_scaled * torch.log(router_score_scaled + 1e-5), dim=-1)
        router_entropy = router_entropy.mean()
        # add chunk sparsification loss
        chunk_loss = ActivationContext.chunk_regularization(router_score_scaled, seq_mask)

        # add load balance
        act_flag = torch.gt(router_score_scaled, 0)
        load = act_flag.sum(dim=0)
        load_mean = load / (torch.sum(act_flag) + 1e-5)
        importance_mean = router_score_scaled.mean(dim=0)
        balance_loss = self.num_expert * torch.sum(importance_mean * load_mean)

        ActivationContext.append_intermediate(router_ori)
        # >>> temporary fix <<<
        l1_loss = ActivationContext.l1_regularization(router_score, self.layer_id)
        transfer_loss = ActivationContext.transfer_regularization(raw_router_score, router_score, segment_shift_indices, self.layer_id, self.gate_policy, raw_router_gate)
        token_balance_loss = ActivationContext.token_balance_regularization(raw_router_score, segment_shift_indices, self.gate_policy)
        ActivationRecorder.record_fp_mlp_label(self.layer_id, router_score)

        router_up_proj = None
        if self.gate_policy in ["gate1", "gate2", "relugate2"]: 
            router_up_proj = router_score.unsqueeze(-1) * self.router_up_proj
        if self.gate_policy == "gate4":
            router_up_proj = self.router_x_up_proj(torch.relu(self.router_x_down_proj(router_input)))
            router_up_proj = router_up_proj.view(batch_size * seq_len, self.num_expert, self.dim_expert)
            router_up_proj = router_score.unsqueeze(-1) * router_up_proj

        x = self.moe_experts(x, router_score, router_up_proj)
        x = x.view(batch_size, seq_len, dim_model)

        ActivationContext.append_output(x)

        return x, l1_loss, balance_loss, router_entropy, transfer_loss, chunk_loss, token_balance_loss
