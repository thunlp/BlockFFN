import torch
import torch.nn.functional as F


class SquaredReLU(torch.nn.Module):
    def forward(self, x):
        return torch.square(F.relu(x))


class NullAct(torch.nn.Module):
    def forward(self, x):
        return x


def get_activation_fn(activate_fn: str):
    if activate_fn == "gelu":
        act = torch.nn.GELU()
    elif activate_fn == "silu":
        act = torch.nn.SiLU()
    elif activate_fn == "relu":
        act = torch.nn.ReLU()
    elif activate_fn == "sqrelu":
        act = SquaredReLU()
    elif activate_fn == "sigmoid":
        act = torch.nn.Sigmoid()
    elif activate_fn == "softmax":
        act = torch.nn.Softmax(dim=-1)
    elif activate_fn == "null":
        act = NullAct()
    else:
        raise NotImplementedError(f"{activate_fn} is not supported")
    return act
