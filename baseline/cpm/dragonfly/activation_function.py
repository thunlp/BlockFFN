import torch
import bmtrain as bmt
import torch.nn.functional as F
from .layer_norm import LayerNorm


class ShiftedReLU(torch.nn.Module):
    def __init__(self, bias: float = 0.) -> None:
        super().__init__()
        self.bias = bias

    def forward(self, x):
        return F.relu(x - self.bias)


class FATReLU(torch.nn.Module):
    def __init__(self, threshold: float = 0.) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        new_x = torch.zeros_like(x)
        mask = torch.ge(x, self.threshold)
        new_x[mask] = x[mask]
        return new_x


class SquaredReLU(torch.nn.Module):
    def forward(self, x):
        return torch.square(F.relu(x))


class NullAct(torch.nn.Module):
    def forward(self, x):
        return x


class AlphaTanh(torch.nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        return self.beta * torch.tanh(self.alpha * x)


class LeCunTanh(torch.nn.Module):
    # equivalent to AlphaTanh(0.6667, 1.7159)
    def forward(self, x):
        return 1.7159 * torch.tanh(x * (2/3))


class ArcSinh(torch.nn.Module):
    def __init__(self, alpha: float = 0.1, beta: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        return self.beta * torch.arcsinh(self.alpha * x)


class DynamicArcsinh(bmt.DistributedModule):
    def __init__(self, init_alpha: float = 0.1, init_beta: float = 1.0) -> None:
        super().__init__()
        self.alpha = bmt.DistributedParameter(
            torch.empty(1, dtype=torch.bfloat16),
            init_method=bmt.ParameterInitializer(torch.nn.init.constant_, val=init_alpha),
        )
        self.beta = None
        if init_beta > 0:
            self.beta = bmt.DistributedParameter(
                torch.empty(1, dtype=torch.bfloat16),
                init_method=bmt.ParameterInitializer(torch.nn.init.constant_, val=init_beta),
            )

    def forward(self, x):
        y = torch.arcsinh(self.alpha * x)
        if self.beta is not None:
            y = self.beta * y
        return y


class DynamicTanh(bmt.DistributedModule):
    def __init__(self, init_alpha: float = 0.5, init_beta: float = -1) -> None:
        super().__init__()
        self.alpha = bmt.DistributedParameter(
            torch.empty(1, dtype=torch.bfloat16),
            init_method=bmt.ParameterInitializer(torch.nn.init.constant_, val=init_alpha),
        )
        self.beta = None
        if init_beta > 0:
            self.beta = bmt.DistributedParameter(
                torch.empty(1, dtype=torch.bfloat16),
                init_method=bmt.ParameterInitializer(torch.nn.init.constant_, val=init_beta),
            )

    def forward(self, x):
        y = torch.tanh(self.alpha * x)
        if self.beta is not None:
            y = self.beta * y
        return y


class ParametricReLU(bmt.DistributedModule):
    def __init__(self, init_alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = bmt.DistributedParameter(
            torch.empty(1, dtype=torch.bfloat16),
            init_method=bmt.ParameterInitializer(torch.nn.init.constant_, val=init_alpha),
        )

    def forward(self, x):
        return F.prelu(x, weight=self.alpha)


class OddSiLU(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x.abs())


class NegSiLU(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(-x.abs())


def get_activation_fn(activate_fn: str, **kwargs):
    if activate_fn == "gelu":
        act = torch.nn.GELU()
    elif activate_fn == "silu":
        act = torch.nn.SiLU()
    elif activate_fn == "norm_silu":
        act = torch.nn.Sequential(
            LayerNorm(dim_norm=kwargs["dim_norm"], dtype=kwargs["dtype"], eps=kwargs["eps"], norm_type="normal"),
            torch.nn.SiLU(),
        )
    elif activate_fn == "fix_norm_silu":
        act = torch.nn.Sequential(
            LayerNorm(dim_norm=kwargs["dim_norm"], dtype=kwargs["dtype"], eps=kwargs["eps"], norm_type="normal", fixed=True),
            torch.nn.SiLU(),
        )
    elif activate_fn == "norm_relu":
        act = torch.nn.Sequential(
            LayerNorm(dim_norm=kwargs["dim_norm"], dtype=kwargs["dtype"], eps=kwargs["eps"], norm_type="normal"),
            torch.nn.ReLU(),
        )
    elif activate_fn == "relu":
        act = torch.nn.ReLU()
    elif activate_fn == "sqrelu":
        act = SquaredReLU()
    elif activate_fn == "sigmoid":
        act = torch.nn.Sigmoid()
    elif activate_fn.startswith("shiftrelu"):
        _, bias = activate_fn.split("_")
        bias = float(bias)
        act = ShiftedReLU(bias=bias)
    elif activate_fn.startswith("fatrelu"):
        _, threshold = activate_fn.split("_")
        threshold = float(threshold)
        act = FATReLU(threshold=threshold)
    elif activate_fn == "softmax":
        act = torch.nn.Softmax(dim=-1)
    elif activate_fn == "null":
        act = NullAct()
    elif activate_fn == "selu":
        act = torch.nn.SELU()
    elif activate_fn == "odd_silu":
        act = OddSiLU()
    elif activate_fn == "neg_silu":
        act = NegSiLU()
    elif activate_fn.startswith("alpha_tanh"):
        tokens = activate_fn.split("_")
        if len(tokens) == 3:
            act = AlphaTanh(alpha=float(tokens[2]))
        else:
            assert len(tokens) == 4
            act = AlphaTanh(alpha=float(tokens[2]), beta=float(tokens[3]))
    elif activate_fn.startswith("dynamic_tanh"):
        tokens = activate_fn.split("_")
        if len(tokens) == 3:
            act = DynamicTanh(init_alpha=float(tokens[2]))
        else:
            assert len(tokens) == 4
            act = DynamicTanh(init_alpha=float(tokens[2]), init_beta=float(tokens[3]))
    elif activate_fn.startswith("leaky_relu"):
        tokens = activate_fn.split("_")
        assert len(tokens) == 3
        act = torch.nn.LeakyReLU(negative_slope=float(tokens[2]))
    elif activate_fn.startswith("parametric_relu"):
        tokens = activate_fn.split("_")
        assert len(tokens) == 3
        act = ParametricReLU(init_alpha=float(tokens[2]))
    elif activate_fn.startswith("arcsinh"):
        tokens = activate_fn.split("_")
        if len(tokens) == 2:
            act = ArcSinh(alpha=float(tokens[1]))
        else:
            assert len(tokens) == 3
            act = ArcSinh(alpha=float(tokens[1]), beta=float(tokens[2]))
    elif activate_fn.startswith("norm_arcsinh"):
        tokens = activate_fn.split("_")
        if len(tokens) == 3:
            arc_sinh = ArcSinh(alpha=float(tokens[2]))
        else:
            assert len(tokens) == 4
            arc_sinh = ArcSinh(alpha=float(tokens[2]), beta=float(tokens[3]))
        act = torch.nn.Sequential(
            LayerNorm(dim_norm=kwargs["dim_norm"], dtype=kwargs["dtype"], eps=kwargs["eps"], norm_type="normal"),
            arc_sinh,
        )
    elif activate_fn.startswith("dynamic_arcsinh"):
        tokens = activate_fn.split("_")
        if len(tokens) == 3:
            act = DynamicArcsinh(init_alpha=float(tokens[2]))
        else:
            assert len(tokens) == 4
            act = DynamicArcsinh(init_alpha=float(tokens[2]), init_beta=float(tokens[3]))
    else:
        raise NotImplementedError(f"{activate_fn} is not supported")
    return act
