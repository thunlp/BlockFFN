import tree
import torch
import bmtrain as bmt
from typing import Optional
import torch.nn.functional as F
from .activation_function import get_activation_fn
from .activation_context import ActivationContext
from fmoe.layers import _fmoe_general_global_forward


class MoELinearActiveGate(bmt.DistributedModule):
    def __init__(self, dim_model, num_expert, activate_fn: str = "relu", activate_kwargs: dict = {},
                 dtype=torch.bfloat16, init_mean=0.0, init_std=0.02):
        super().__init__()
        self.w_gate = bmt.DistributedParameter(
            torch.empty((num_expert, dim_model), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
        )
        self.act = get_activation_fn(activate_fn)

    def forward(self, x):
        """
        Args: x (:obj:`torch.Tensor` of shape ``(batch * seq_len, dim_in)``)
        Return: gating_scores (:obj`torch.Tensor` of shape ``(batch * seq_len, expert_num)``)
        """
        raw_score = F.linear(x, self.w_gate)
        return raw_score, self.act(raw_score)


class MoEUpDownExperts(bmt.DistributedModule):
    def __init__(self, dim_model: int, dim_expert: int, num_expert: int, expert_gated: bool,
                 activate_fn: str = "silu", activate_kwargs: dict = {}, use_linear: bool = False,
                 dropout_p: Optional[float] = None, dtype: torch.dtype = torch.bfloat16,
                 init_mean: float = 0.0, init_std: float = 0.02):
        super().__init__()

        # MoE DenseGatedACT
        if use_linear:
            if expert_gated:
                self.moe_w_gate = bmt.DistributedParameter(
                    torch.empty((num_expert * dim_expert, dim_model), dtype=dtype),
                    init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
                )
            self.moe_w_in = bmt.DistributedParameter(
                torch.empty((num_expert * dim_expert, dim_model), dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            )
            self.moe_w_out = bmt.DistributedParameter(
                torch.empty((dim_model, num_expert * dim_expert), dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            )
        else:
            if expert_gated:
                self.moe_w_gate = bmt.DistributedParameter(
                    torch.empty((num_expert, dim_expert, dim_model), dtype=dtype),
                    init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
                )
            self.moe_w_in = bmt.DistributedParameter(
                torch.empty((num_expert, dim_expert, dim_model), dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            )
            self.moe_w_out = bmt.DistributedParameter(
                torch.empty((num_expert, dim_model, dim_expert), dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            )
        self.act = get_activation_fn(activate_fn)
        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None
        self.num_expert, self.dim_expert, self.dim_model, self.use_linear, self.expert_gated = num_expert, dim_expert, dim_model, use_linear, expert_gated

        if not self.use_linear:
            raise NotImplementedError("block_linear if more recommended for efficiency")

    def forward(self, x, router_score, router_up_proj=None):
        seq_len = router_score.shape[0]
        if self.expert_gated:
            if self.use_linear:
                x_score = F.linear(x, self.moe_w_gate)
                x_in = F.linear(x, self.moe_w_in)
            else:
                x_score = torch.matmul(x, self.moe_w_gate.transpose(1, 2))
                x_in = torch.matmul(x, self.moe_w_in.transpose(1, 2))
            x_in = self.act(x_score) * x_in
        else:
            if self.use_linear:
                x_in = F.linear(x, self.moe_w_in)
            else:
                x_in = torch.matmul(x, self.moe_w_in.transpose(1, 2))
            x_in = self.act(x_in)
        if self.dropout is not None:
            x_in = self.dropout(x_in)
        ActivationContext.stat_moe_intermediate_activation(x_in)
        if self.use_linear:
            if router_up_proj is not None:
                assert router_up_proj.shape[0] == seq_len
                scored_x_in = x_in.view(seq_len, self.num_expert, self.dim_expert) * router_up_proj
            else:
                scored_x_in = x_in.view(seq_len, self.num_expert, self.dim_expert) * router_score.unsqueeze(-1)
            x_out = F.linear(scored_x_in.view(seq_len, self.num_expert * self.dim_expert), self.moe_w_out)
        else:
            if router_up_proj is not None:
                assert router_up_proj.shape[0] == seq_len
                scored_x_in = x_in * router_up_proj.transpose(0, 1)
            else:
                scored_x_in = x_in * router_score.T.unsqueeze(-1)
            # [num_expert-e, seq_len-s, dim_expert-d] @ [num_expert-e, dim_model-m, dim_expert-d]
            x_out = torch.einsum("esd,emd->sm", scored_x_in, self.moe_w_out)
        return x_out


class TopkRouter(bmt.DistributedModule):
    """
    Select top_k expert each time, with a learnable gate_network controlling expert scores.
    https://arxiv.org/pdf/2101.03961.pdf
    """
    def __init__(self,
        dim_model: int,
        num_experts: int,
        top_k: int,
        dtype: torch.dtype,
        init_mean: float = 0,
        init_std: float = 0.01
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.weight = bmt.DistributedParameter(
            torch.empty((num_experts, dim_model), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std)
        )

    def forward(self, x: torch.Tensor):
        # [bs, seq_len, hidden_size]
        x = x.view(-1, x.shape[-1])
        # [bs * seq_len, hidden_size]
        scores = F.linear(x, self.weight)
        # [bs * seq_len, num_experts]
        scores_prob = F.softmax(scores, dim=-1, dtype=torch.float32)
        # [bs * seq_len, num_experts]
        expert_weights, expert_indices = torch.topk(scores_prob, self.top_k, dim=-1)
        # NOTE: mixtral use topk norm, deepseek-moe does't norm topk probs
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        # [bs * seq_len, topk], [bs * seq_len, topk]

        # calculate load balancing loss
        token_num = x.shape[0]
        load = expert_indices.view(-1).bincount(minlength=self.num_experts)
        load_mean = load / (token_num * self.top_k)
        importance_mean = scores_prob.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(importance_mean * load_mean)

        return {
            "topk_indices": expert_indices,
            "topk_scores": expert_weights.to(x.dtype),
            "load": load_mean,
            "balance_loss": balance_loss
        }


class FastTopkCalculator:
    def __init__(self, num_experts, top_k):
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, hidden_states, topk_indices, topk_weights, experts, **kwargs):
        batch_size, seq_len, dim = hidden_states.shape
        hidden_states = hidden_states.view(batch_size * seq_len, dim)
        fwd = _fmoe_general_global_forward(
            hidden_states, topk_indices, experts, self.num_experts, world_size=1, experts=experts,
        )

        def view_func(tensor):
            n_dim = tensor.shape[-1]
            tensor = tensor.view(-1, self.top_k, n_dim)
            return tensor

        moe_output = tree.map_structure(view_func, fwd)
        topk_weights = topk_weights.unsqueeze(1)

        def bmm_func(tensor):
            n_dim = tensor.shape[-1]
            tensor = torch.bmm(topk_weights, tensor).reshape(-1, n_dim)
            return tensor

        moe_output = tree.map_structure(bmm_func, moe_output)
        moe_output = moe_output.view(batch_size, seq_len, -1)
        return moe_output
