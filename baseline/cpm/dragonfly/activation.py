import torch
from typing import List, Tuple


class MoEAuxLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that compute and scales the grad for auxiliary loss."""

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        """Preserve the aux_loss by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            aux_loss (torch.Tensor): The auxiliary loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for auxiliary loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled auxiliary loss
                                               gradient.
        """
        (aux_loss,) = ctx.saved_tensors
        aux_loss_backward_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the aux loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        MoEAuxLossAutoScaler.main_loss_backward_scale = scale


class ActivationContext:
    _activation_num: int = 0
    _total_num: int = 0
    _inference_mode: bool = False
    _stat_act_window_sizes: List[int] = None
    _stat_window_activation_nums: List[int] = None
    _stat_window_total_nums: List[int] = None
    _moe_mid_activation_num: int = 0
    _moe_mid_total_num: int = 0

    # for ReLUMoE
    _layer_num: int = None
    _remoe_l1_reg_init: float = None
    _remoe_l1_reg_coef: float = None
    _remoe_l1_reg_coef_multiplier: float = None

    @classmethod
    def set_inference_mode(cls, inference_mode: bool):
        cls._inference_mode = inference_mode

    @classmethod
    def set_remoe_l1_reg_info(cls, layer_num: int, l1_reg_coef_init: float, l1_reg_coef_multiplier: float, l1_reg_coef_resume: float):
        cls._layer_num = layer_num
        cls._remoe_l1_reg_init = l1_reg_coef_init
        cls._remoe_l1_reg_coef = l1_reg_coef_resume if l1_reg_coef_resume > 0 else l1_reg_coef_init
        cls._remoe_l1_reg_coef_multiplier = l1_reg_coef_multiplier

    @classmethod
    def get_remoe_l1_reg_coef(cls):
        return cls._remoe_l1_reg_coef

    @classmethod
    def set_stat_act_window_sizes(cls, stat_act_window_sizes: str):
        cls._stat_act_window_sizes = [int(s) for s in stat_act_window_sizes.split(",")]
        cls._stat_window_activation_nums = [0] * len(cls._stat_act_window_sizes)
        cls._stat_window_total_nums = [0] * len(cls._stat_act_window_sizes)

    @classmethod
    @torch.no_grad()
    def stat_act_rate(cls, topk_indices, num_shared_experts, num_routed_experts, seq_mask):
        seq_len, max_act = topk_indices.shape
        with torch.no_grad():
            act_num = torch.sum(torch.ne(topk_indices, -1)).item()
            act_num += seq_len * num_shared_experts
            act_total = seq_len * (num_shared_experts + num_routed_experts)
        cls._activation_num += act_num
        cls._total_num += act_total

        if cls._inference_mode:
            stat_indices = topk_indices[seq_mask]
            valid_seq_len = stat_indices.shape[0]
            for idx, s in enumerate(cls._stat_act_window_sizes):
                truncate_seq_len = valid_seq_len
                truncate_indices = stat_indices
                if truncate_seq_len % s != 0:
                    truncate_seq_len -= truncate_seq_len % s
                    truncate_indices = stat_indices[:truncate_seq_len]
                seg_num = truncate_seq_len // s
                loc_indices = truncate_indices.view(seg_num, s, max_act).view(seg_num, -1).tolist()
                for seg in loc_indices:
                    cls._stat_window_activation_nums[idx] += len(set(seg) - {-1}) + num_shared_experts
                cls._stat_window_total_nums[idx] += seg_num * (num_shared_experts + num_routed_experts)

    @classmethod
    def get_clear_act(cls):
        activation_num, total_num = cls._activation_num, cls._total_num
        cls._activation_num, cls._total_num = 0, 0
        return activation_num, total_num

    @classmethod
    def stat_moe_intermediate_activation(cls, activation: torch.Tensor):
        if cls._inference_mode or activation.requires_grad:
            with torch.no_grad():
                loc_act, loc_tot = torch.sum(torch.gt(activation, 0)).item(), activation.numel()
                cls._moe_mid_activation_num += loc_act
                cls._moe_mid_total_num += loc_tot

    @classmethod
    def get_clear_moe_intermediate_activation(cls):
        mid_activation_num, mid_total_num = cls._moe_mid_activation_num, cls._moe_mid_total_num
        cls._moe_mid_activation_num, cls._moe_mid_total_num = 0, 0
        return mid_activation_num, mid_total_num

    @classmethod
    def get_clear_transfer_loss(cls):
        assert len(cls._stat_act_window_sizes) == len(cls._stat_window_activation_nums) == len(
            cls._stat_window_total_nums)
        window_sizes, activation_nums, total_nums = \
            cls._stat_act_window_sizes[:], cls._stat_window_activation_nums, cls._stat_window_total_nums
        cls._stat_window_activation_nums = [0] * len(cls._stat_act_window_sizes)
        cls._stat_window_total_nums = [0] * len(cls._stat_act_window_sizes)
        return window_sizes, activation_nums, total_nums

    @classmethod
    def step_l1_ref_coef(cls, cur_act_rate: float, target_act_rate: float) -> None:
        l1_coef, multiplier = cls._remoe_l1_reg_coef, cls._remoe_l1_reg_coef_multiplier
        assert l1_coef is not None and multiplier is not None
        if l1_coef <= 0 or multiplier <= 0:
            return
        if cur_act_rate > target_act_rate:
            cls._remoe_l1_reg_coef *= multiplier
        else:
            cls._remoe_l1_reg_coef = max(cls._remoe_l1_reg_coef / multiplier, cls._remoe_l1_reg_init)

    @classmethod
    def l1_reg_load_balancing(cls, probs: torch.Tensor, routing_map: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l1_coef = cls._remoe_l1_reg_coef
        tokens_per_expert = routing_map.sum(dim=0)
        assert l1_coef is not None
        if l1_coef <= 0:
            return probs, torch.tensor(0.).to(probs), tokens_per_expert

        num_tokens, num_experts = probs.shape
        # L1 regularization with load balancing shares the same formula with switch load balancing loss:
        # l1_reg = sum((probs_per_expert/num_tokens) * (tokens_per_expert/(num_tokens*topk))) * num_experts * l1_coef
        aggregated_probs_per_expert = probs.sum(dim=0)
        l1_reg = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
                num_experts * l1_coef / (num_tokens * num_tokens * top_k)
        )
        assert cls._layer_num is not None
        probs = MoEAuxLossAutoScaler.apply(probs, l1_reg / cls._layer_num)
        return probs, l1_reg.detach().clone(), tokens_per_expert
