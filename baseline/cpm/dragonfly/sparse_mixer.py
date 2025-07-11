import torch
import bmtrain as bmt
import torch.nn.functional as F
from typing import Dict, Callable
from torch.distributions import Uniform

uniform_map: Dict[torch.device, Callable] = {}


def multiplicative_jitter(inputs, epsilon, training):
    """
    inputs multiply by a uniform distribution noise, which is called jitter
    """
    if epsilon == 0 or not training:
        return inputs

    uniform = uniform_map.get(inputs.device)

    if uniform is None:
        uniform = Uniform(low=torch.tensor(1.0 - epsilon, device=inputs.device, dtype=inputs.dtype),
                          high=torch.tensor(1.0 + epsilon, device=inputs.device, dtype=inputs.dtype)
                          ).rsample
        uniform_map[inputs.device] = uniform
    return inputs * uniform(inputs.shape)


class CoreV2(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            scores: torch.Tensor,
            multiplier: torch.Tensor,
            selected_experts: torch.Tensor,
            masked_gates: torch.Tensor,
            mask_for_one: torch.Tensor,
    ):
        ctx.save_for_backward(multiplier, selected_experts, masked_gates)
        return multiplier * mask_for_one

    @staticmethod
    def backward(
            ctx,
            grad_at_output: torch.Tensor,
    ):
        multiplier, selected_experts, masked_gates = ctx.saved_tensors

        grad_at_output = grad_at_output * multiplier

        grad_at_scores_expaned = masked_gates * grad_at_output.mul(-1)
        grad_at_scores_expaned.scatter_add_(
            dim=-1,
            index=selected_experts,
            src=grad_at_output,
        )

        return (
            grad_at_scores_expaned,
            None,
            None,
            None,
            None,
        )


def sparse_mixer_v2_routing(scores, top_k, jitter_eps, training):
    original_scores = scores
    original_gates = torch.softmax(scores, dim=-1)
    selected_experts, multiplier = None, None

    ################ iterative sampling ################
    for eid in range(top_k):
        # masked out the precious expert
        if selected_experts is not None:
            scores = torch.scatter(
                original_scores,
                -1,
                selected_experts,
                float('-inf'),
            )

        ################ select the eid-th expert ################
        with torch.no_grad():
            # compute mask for sparsity
            mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)
            factor = original_scores.abs().clamp(min=mask_logits_threshold)
            mask_logits_threshold = ((mask_logits_threshold - original_scores) / factor) > (2 * jitter_eps)

        # apply mask
        masked_gates = scores.masked_fill(mask_logits_threshold, float('-inf'))
        if training:
            selected_experts_eid = (
                    masked_gates - torch.empty_like(masked_gates,
                                                    memory_format=torch.legacy_contiguous_format).exponential_().log()
            ).max(dim=-1)[1].unsqueeze(-1)  # gumbel sampling, more robust than the multinomial method
        else:
            selected_experts_eid = max_ind

        # compute scores for gradients
        masked_gates = torch.softmax(masked_gates, dim=-1)

        # compute midpoint mask
        max_scores, max_ind = masked_gates.max(dim=-1, keepdim=True)
        mask_for_one = torch.logical_or(
            torch.eq(selected_experts_eid, max_ind),
            torch.rand_like(max_scores) > 0.75  # Heun's third-order method: f(x) - f(0) = .25 f'(x) + .75 f'(x/3.)
        )
        # 1 -> 1.0 & 0 -> 1./3: lambda x: (x + 0.5) / 1.5
        mask_for_one = torch.add(0.3333, mask_for_one, alpha=0.6667).type_as(masked_gates)

        multiplier_o = masked_gates.gather(dim=-1, index=selected_experts_eid)
        multiplier_eid = CoreV2.apply(
            original_scores,
            multiplier_o,
            selected_experts_eid,
            masked_gates,
            mask_for_one,
        )

        if multiplier is None:
            multiplier = multiplier_eid
            assert selected_experts is None
            selected_experts = selected_experts_eid
        else:
            multiplier = torch.concat((multiplier, multiplier_eid), dim=-1)
            assert selected_experts is not None
            selected_experts = torch.concat((selected_experts, selected_experts_eid), dim=-1)

    return (
        multiplier,
        original_gates,
        selected_experts,
    )


class SparseMixerV2(bmt.DistributedModule):
    def __init__(
        self,
        dim_model: int,
        num_experts: int,
        top_k: int,
        dtype: torch.dtype,
        init_mean: float = 0,
        init_std: float = 0.01,
        jitter_eps: float = 0.1,
    ):
        super(SparseMixerV2, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_eps = jitter_eps

        self.weight = bmt.DistributedParameter(
            torch.empty((num_experts, dim_model), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std)
        )

    def forward(self, x: torch.Tensor):
        x = x.view(-1, x.shape[-1])
        logits = F.linear(x, self.weight)
        multiplier, original_gates, selected_experts = sparse_mixer_v2_routing(logits, self.top_k, self.jitter_eps, self.training)

        num_tokens = F.one_hot(selected_experts.squeeze(-1), self.num_experts).gt(0).sum(0)
        load_mean = num_tokens / (num_tokens.sum(0, keepdim=True) + 1e-6)
        p = logits.softmax(dim=-1)
        importance_mean = p.view(-1, self.num_experts).mean(0)
        balance_loss = self.num_experts * torch.sum(importance_mean * load_mean)

        return {
            "topk_indices": selected_experts,
            "topk_scores": multiplier,
            "load": load_mean,
            "balance_loss": balance_loss,
            "router_entropy": torch.tensor(0.).to(x),  # not implemented
        }
