import torch
import bmtrain as bmt
from typing import Union


# @torch.jit.script
def rms_layernorm(hidden: torch.Tensor, weight: Union[torch.Tensor, float], eps: float):
    old_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden, hidden * weight


# @torch.jit.script
def normal_layernorm(hidden: torch.Tensor, weight: Union[torch.Tensor, float], eps: float):
    old_dtype = hidden.dtype
    hidden = hidden - torch.mean(hidden, dim=hidden.ndim-1, keepdim=True)
    variance = hidden.to(torch.float32).pow(2).mean(dim=hidden.ndim-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden, hidden * weight


class LayerNorm(bmt.DistributedModule):
    """RMS LayerNorm"""

    def __init__(
        self,
        dim_norm: int,
        dtype: torch.dtype = torch.bfloat16,
        eps: float = 1e-6,
        init_var: float = 1.0,
        norm_type: str = "rms",
        fixed: bool = False,
    ):
        super().__init__()

        self.eps = eps
        self.dim_norm = dim_norm
        self.fixed = fixed
        if self.fixed:
            self.weight = init_var
        else:
            self.weight = bmt.DistributedParameter(torch.full((dim_norm,), init_var, dtype=dtype))
        assert norm_type in ["rms", "normal", "simple"]
        self.norm_type = norm_type

    def forward(self, x: torch.Tensor, output_hidden: bool = False):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch_size, seq_len, dim_norm)``): Input tensor that need to be normalized.
        Return:
            :obj:`torch.Tensor` of shape ``(batch_size, seq_len, dim_norm)``: The layernorm output.
        """  # noqa: E501
        assert x.size(-1) == self.dim_norm
        if self.norm_type == "rms":
            res, weighted_res = rms_layernorm(x, self.weight, self.eps)
        elif self.norm_type == "normal":
            res, weighted_res = normal_layernorm(x, self.weight, self.eps)
        else:
            res, weighted_res = x, x * self.weight
        return (res, weighted_res) if output_hidden else weighted_res
