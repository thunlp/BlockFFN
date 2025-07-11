import torch
import bmtrain as bmt
import math
from .activation_function import get_activation_fn
from megablocks import stk, sparse_act
import torch.nn.functional as F


ACT_MAP = {
    "relu": lambda x: sparse_act(x, F.relu(x.data)),
    "gelu": lambda x: sparse_act(x, F.gelu(x.data, approximate="tanh")),
    "relu2": lambda x: sparse_act(x, torch.square(F.relu(x.data))),
    "silu": lambda x: sparse_act(x, F.silu(x.data)),
}


class MegaMoELinearActiveGate(bmt.DistributedModule):
    def __init__(self, dim_model, num_expert, activate_fn: str = "relu",
                 dtype=torch.bfloat16, init_mean=0.0, init_std=0.02, scale=False, scale_width=1.0, tp=0):
        super().__init__()
        _std = init_std / math.sqrt(scale_width) if scale else init_std
        if tp:
            self.w_gate = bmt.DistributedParameter(
                torch.empty((num_expert, dim_model), dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=_std),
                tp_split_dim=1, tp_mode=True
            )
        else:
            self.w_gate = bmt.DistributedParameter(
                torch.empty((num_expert, dim_model), dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=_std),
            )
        self.act = get_activation_fn(activate_fn)

    def forward(self, x):
        """
        Args: x (:obj:`torch.Tensor` of shape ``(batch * seq_len, dim_in)``)
        Return: gating_scores (:obj`torch.Tensor` of shape ``(batch * seq_len, expert_num)``)
        """
        return self.act(F.linear(x, self.w_gate))


class BlockSparseMLP(bmt.DistributedModule):
    def __init__(
            self,
            dim_model: int,
            dim_ff: int,
            num_experts: int,
            activate_fn: str = "relu",
            dtype: torch.dtype = torch.float32,
            init_mean: float = 0.0,
            init_std: float = 0.02,
            scale=False,
            scale_width=1.0,
            tp=0
    ):
        super().__init__()
        _std = init_std / math.sqrt(scale_width) if scale else init_std
        if tp:
            self.w1 = bmt.DistributedParameter(
                torch.empty(num_experts * dim_ff, dim_model, dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
                tp_split_dim=0, tp_mode=True
            )
            self.w2 = bmt.DistributedParameter(
                torch.empty(num_experts * dim_ff, dim_model, dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
                tp_split_dim=0, tp_mode=True
            )
        else:
            self.w1 = bmt.DistributedParameter(
                torch.empty(num_experts * dim_ff, dim_model, dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            )
            self.w2 = bmt.DistributedParameter(
                torch.empty(num_experts * dim_ff, dim_model, dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            )
        self.act = ACT_MAP[activate_fn]

    def forward(self, x, topo):
        x = stk.Matrix(
            topo.size(),
            self.act(stk.ops.sdd(x, self.w1.t(), topo)).data,
            topo.row_indices,
            topo.column_indices,
            topo.offsets,
            topo.column_indices_t,
            topo.offsets_t,
            topo.block_offsets_t
        )
        x = stk.ops.dsd(x, self.w2)
        return x
