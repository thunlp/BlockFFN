import torch
import bmtrain as bmt
from fmoe.linear import MOELinear
import torch.nn.functional as F


def Linear(*args, **kwargs):
    tp = kwargs.pop("tp", 0)
    num_experts = kwargs.pop("num_experts", -1)
    if num_experts > 0:
        assert tp == 0
        kwargs["num_experts"] = num_experts
        return MoELinearExperts(*args, **kwargs)
    if tp == 0:
        return NormalLinear(*args, **kwargs)
    if tp == 1:
        return ColumnParallelLinear(*args, **kwargs)
    if tp == 2:
        return RowParallelLinear(*args, **kwargs)


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
        self.num_experts = num_experts

        # TODO:init
        # init_std = 1 / ((dim_in + dim_out) ** 0.5)

        self.weight = bmt.DistributedParameter(
            torch.empty((num_experts, dim_out, dim_in), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
        )

    def forward(self, x: torch.Tensor, fwd_expert_count: torch.Tensor):
        x = MOELinear.apply(x, fwd_expert_count, self.weight, None)
        return x


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

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501

        x = F.linear(x, self.weight, None)

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
