import math
import torch
import bmtrain as bmt
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union


def retain_backward(optim_manager: bmt.optim.OptimManager, loss: torch.Tensor):
    if not (torch.isnan(loss) or torch.isinf(loss)):
        loss = optim_manager.scale_loss(loss)
        loss.backward(retain_graph=True)
    current_stream = torch.cuda.current_stream()
    current_stream.wait_stream(bmt.config['load_stream'])


def stable_bce(A, B, C, D, eps=1e-12):
    # Compute log(sigmoid(A) * sigmoid(B)) using stable terms
    log_p = - F.softplus(-A) - F.softplus(-B)  # Sum of logs

    # Compute log(1 - sigmoid(A)*sigmoid(B)) using logsumexp
    log_1_minus_p = torch.logsumexp(torch.stack([-A, -B, -A-B], dim=0), dim=0) - (F.softplus(-A) + F.softplus(-B))

    # Compute target y = sigmoid(C) * sigmoid(D) with clamping
    y = torch.sigmoid(C) * torch.sigmoid(D)
    y = y.clamp(eps, 1 - eps)  # Avoid 0 and 1

    # Calculate BCE loss components
    loss = - (y * log_p + (1 - y) * log_1_minus_p)
    return loss.mean()


"""
def log1mexp(x):
    mask = x < -0.693147
    return torch.where(mask, torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x)))


def stable_bce2(A, B, C, D):
    logsigmoid = torch.nn.LogSigmoid()
    # Predicted log-probability
    log_p_pred = logsigmoid(A) + logsigmoid(B)

    # Log(1 - p_pred)
    log_1_minus_p_pred = log1mexp(log_p_pred)

    # Target probability
    log_p_target = logsigmoid(C) + logsigmoid(D)
    p_target = torch.exp(log_p_target)

    # BCE loss
    loss = - (p_target * log_p_pred + (1 - p_target) * log_1_minus_p_pred)
    return loss.mean()
"""


class ActivationContext:

    _l1_lambda_option: str = ""
    _l1_lambda: float = -1
    _end_l1_lambda: float = -1
    _cosine_step: int = 0
    _end_cosine_step: int = 0
    _accu_iters: int = 1
    _l1_losses: List[float] = []
    _activation_num: int = 0
    _total_num: int = 0
    _layer_stat_enabled: bool = False
    _layer_activation_nums: List[int] = []
    _layer_total_nums: List[int] = []
    _expert_activation_num: List[List[int]] = []
    _expert_total_num: List[List[int]] = []

    _moe_mid_activation_num: int = 0
    _moe_mid_total_num: int = 0

    _transfer_lambda: float = -1
    _sigmoid_steep: float = 100.
    _stat_act_window_sizes: List[int] = None
    _stat_window_activation_nums: List[int] = None
    _stat_window_total_nums: List[int] = None
    _transfer_losses: List[float] = []
    _chunk_regularization_enabled: bool = False
    _chunk_regularization_length: int = -1

    _token_balance_factor: float = -1

    _optim_manager: Optional[bmt.optim.OptimManager] = None

    _inference_mode: bool = False
    _layer_to_task_tokens: Optional[torch.Tensor] = None
    _layer_to_task_activations: Optional[torch.Tensor] = None
    _stat_task_ids: Optional[torch.Tensor] = None

    _record_flag: bool = False
    _input_list: List[torch.Tensor] = []
    _output_list: List[torch.Tensor] = []

    _intermediate_flag: bool = False
    _intermediate_list: List[torch.Tensor] = []

    def __new__(cls, *args, **kwargs):
        raise RuntimeError("ActivationContext object is not allowed to be created!")

    @classmethod
    def set_optim_manager(cls, optim_manager: bmt.optim.OptimManager):
        cls._optim_manager = optim_manager

    @classmethod
    def set_l1_lambda_option(cls, l1_lambda_option: str):
        cls._l1_lambda_option = l1_lambda_option

    @classmethod
    def set_l1_lambda(cls, l1_lambda: float):
        cls._l1_lambda = l1_lambda

    @classmethod
    def set_end_l1_lambda(cls, end_l1_lambda: float):
        cls._end_l1_lambda = end_l1_lambda

    @classmethod
    def set_transfer_lambda(cls, transfer_lambda: float):
        cls._transfer_lambda = transfer_lambda

    @classmethod
    def get_current_transfer_lambda(cls) -> float:
        return cls._transfer_lambda

    @classmethod
    def set_token_balance_factor(cls, token_balance_factor: float):
        cls._token_balance_factor = token_balance_factor

    @classmethod
    def set_sigmoid_steep(cls, sigmoid_steep: float):
        cls._sigmoid_steep = sigmoid_steep

    @classmethod
    def set_stat_act_window_sizes(cls, stat_act_window_sizes: str):
        cls._stat_act_window_sizes = [int(s) for s in stat_act_window_sizes.split(",")]
        cls._stat_window_activation_nums = [0] * len(cls._stat_act_window_sizes)
        cls._stat_window_total_nums = [0] * len(cls._stat_act_window_sizes)

    @classmethod
    def set_cosine_step(cls, step: int):
        cls._cosine_step = step

    @classmethod
    def get_cosine_step(cls) -> int:
        return cls._cosine_step

    @classmethod
    def set_end_cosine_step(cls, step: int):
        cls._end_cosine_step = step

    @classmethod
    def set_inference_mode(cls, inference_mode: bool):
        cls._inference_mode = inference_mode

    @classmethod
    def enable_task_act_stat(cls, num_layers: int, task_num: int):
        cls._layer_to_task_tokens = torch.zeros(num_layers, task_num, dtype=torch.long, device="cuda")
        cls._layer_to_task_activations = torch.zeros(num_layers, task_num, dtype=torch.float32, device="cuda")

    @classmethod
    def get_task_act_stat(cls) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return cls._layer_to_task_tokens, cls._layer_to_task_activations

    @classmethod
    def set_stat_task_ids(cls, stat_task_ids: torch.Tensor):
        cls._stat_task_ids = stat_task_ids

    @classmethod
    def unset_stat_task_ids(cls):
        cls._stat_task_ids = None

    @classmethod
    def set_accu_iters(cls, accu_iters: int):
        cls._accu_iters = accu_iters

    @classmethod
    def set_layer_stat_enabled(cls, layer_stat_enabled: bool):
        cls._layer_stat_enabled = layer_stat_enabled

    @classmethod
    def set_chunk_regularization_enabled(cls, chunk_regularization_enabled: bool):
        cls._chunk_regularization_enabled = chunk_regularization_enabled

    @classmethod
    def set_chunk_regularization_length(cls, chunk_regularization_length: int):
        cls._chunk_regularization_length = chunk_regularization_length

    @classmethod
    def get_current_l1_lambda(cls) -> float:
        if cls._l1_lambda_option == "fixed":
            return cls._l1_lambda
        elif cls._l1_lambda_option == "cosine":
            assert 0 <= cls._l1_lambda < cls._end_l1_lambda, f"[{cls._l1_lambda} | {cls._end_l1_lambda}]"
            assert cls._cosine_step >= 0 and cls._end_cosine_step > 0, f"{cls._cosine_step} | {cls._end_cosine_step}"
            if cls._cosine_step >= cls._end_cosine_step:
                return cls._end_l1_lambda
            # cls._cosine_step < cls._end_cosine_step
            progress = cls._cosine_step / cls._end_cosine_step
            amplitude = (cls._end_l1_lambda - cls._l1_lambda) / 2
            l1_lambda = cls._l1_lambda + amplitude * (math.sin(- math.pi / 2 + progress * math.pi) + 1)
            return l1_lambda
        else:
            raise NotImplementedError(f"Invalid l1_lambda_option: {cls._l1_lambda_option}")

    @classmethod
    def l1_regularization(cls, activation: torch.Tensor, layer_id: int):
        l1_lambda = cls.get_current_l1_lambda()
        accu_loss = 0.
        if l1_lambda > 0:
            # loss = torch.norm(activation, p=1) * 1e-8 * l1_lambda
            loss = torch.norm(activation, p=1) * 1e-8 * 1.0
            accu_loss = loss / cls._accu_iters
            # retain_backward(cls._optim_manager, accu_loss)
            cls._l1_losses.append(loss.item())
        if cls._inference_mode or activation.requires_grad:
            with torch.no_grad():
                loc_act, loc_tot = torch.sum(torch.ne(activation, 0)).item(), activation.numel()
                cls._activation_num += loc_act
                cls._total_num += loc_tot
                if cls._layer_stat_enabled:
                    if layer_id >= len(cls._layer_activation_nums):
                        assert layer_id == len(cls._layer_activation_nums)
                        assert len(cls._layer_activation_nums) == len(cls._layer_total_nums)
                        cls._layer_activation_nums.append(0)
                        cls._layer_total_nums.append(0)
                    cls._layer_activation_nums[layer_id] += loc_act
                    cls._layer_total_nums[layer_id] += loc_tot
        if cls._layer_to_task_activations is not None and (cls._inference_mode or activation.requires_grad):
            with torch.no_grad():
                assert cls._stat_task_ids is not None
                num_layers, task_num = cls._layer_to_task_activations.shape
                task_ids = cls._stat_task_ids + 1  # -1 for invalid tokens
                task_num = task_num + 1
                task_ids = task_ids.view(-1).to(torch.long)
                dim_ff = activation.shape[-1]
                activation = (torch.sum(torch.ne(activation, 0), dim=-1).view(-1) / dim_ff).to(torch.float32)
                seq_len = activation.shape[0]

                one_hot_target = torch.zeros((task_num, *activation.shape), dtype=torch.float32, device="cuda")
                assert one_hot_target.ndim == 2
                one_hot_target[task_ids, torch.arange(seq_len)] = 1
                task_activations = activation.unsqueeze(0) * one_hot_target

                task_avg_token = one_hot_target.to(torch.long).sum(dim=1)[1:]
                task_sum_activations = task_activations.sum(dim=1)[1:] / (task_avg_token + 1e-10)
                cls._layer_to_task_activations[layer_id] += \
                    (task_sum_activations - cls._layer_to_task_activations[layer_id]) * \
                    (task_avg_token / (cls._layer_to_task_tokens[layer_id] + task_avg_token + 1e-10))
                cls._layer_to_task_tokens[layer_id] += task_avg_token
                if torch.sum((torch.isnan(cls._layer_to_task_tokens))) + \
                        torch.sum(torch.isinf(cls._layer_to_task_tokens)) > 0:
                    raise ValueError("Nan value in _layer_to_task_tokens detected!!!")
        return accu_loss

    @classmethod
    def stat_moe_intermediate_activation(cls, activation: torch.Tensor):
        if cls._inference_mode or activation.requires_grad:
            with torch.no_grad():
                loc_act, loc_tot = torch.sum(torch.gt(activation, 0)).item(), activation.numel()
                cls._moe_mid_activation_num += loc_act
                cls._moe_mid_total_num += loc_tot

    @classmethod
    def chunk_regularization(cls, activation: torch.Tensor, seq_mask: torch.Tensor):
        if not cls._chunk_regularization_enabled:
            return 0.
        seq_len, expert_num = activation.shape
        assert seq_mask.shape == (seq_len,)
        chunk_size = cls._chunk_regularization_length
        assert seq_len % chunk_size == 0
        activation_group = activation.view(seq_len // chunk_size, chunk_size, expert_num)
        seq_mask_group = seq_mask.view(seq_len // chunk_size, chunk_size, 1)
        activation_group *= seq_mask_group

        # standard implementation, sum[ln(1-p)]
        activation_group = torch.sum(torch.log(1 - activation_group + 1e-5), dim=1)
        # another simplified implementation, sum(-p)
        # activation_group = - torch.sum(activation_group, dim=1)

        chunk_loss = 1 - torch.mean(torch.exp(activation_group))
        chunk_loss /= cls._accu_iters
        return chunk_loss

    @classmethod
    def get_clear_loss(cls, accumulate: bool = False, output_layer: bool = False) \
            -> Union[Tuple[float, int, int], Tuple[float, int, int, List[int], List[int]]]:
        total_loss = 0.
        for loss in cls._l1_losses:
            total_loss += loss
        cls._l1_losses = []
        activation_num, total_num = cls._activation_num, cls._total_num
        layer_activation_nums, layer_total_nums = cls._layer_activation_nums[:], cls._layer_total_nums[:]
        if not accumulate:
            cls._activation_num, cls._total_num = 0, 0
            cls._layer_activation_nums, cls._layer_total_nums = [], []
        if output_layer:
            return total_loss, activation_num, total_num, layer_activation_nums, layer_total_nums
        else:
            return total_loss, activation_num, total_num

    @classmethod
    def get_clear_moe_intermediate_activation(cls):
        mid_activation_num, mid_total_num = cls._moe_mid_activation_num, cls._moe_mid_total_num
        cls._moe_mid_activation_num, cls._moe_mid_total_num = 0, 0
        return mid_activation_num, mid_total_num

    @classmethod
    @torch.no_grad()
    def stat_window_activation(cls, router_score: torch.Tensor, valid_loss_mask: torch.Tensor = None):
        num_experts = router_score.shape[-1]
        router_score = router_score.view(-1, num_experts)
        stat_router_score = router_score[valid_loss_mask] if valid_loss_mask is not None else router_score
        router_score_mask = torch.ne(stat_router_score, 0)
        valid_seq_len = stat_router_score.shape[0]
        for idx, s in enumerate(cls._stat_act_window_sizes):
            truncate_seq_len = valid_seq_len
            truncate_mask = router_score_mask
            if truncate_seq_len % s != 0:
                truncate_seq_len -= truncate_seq_len % s
                truncate_mask = truncate_mask[:truncate_seq_len]
            seg_num = truncate_seq_len // s
            loc_mask = truncate_mask.view(seg_num, s, num_experts)
            loc_mask = torch.any(loc_mask, dim=1)  # [seg_num, num_experts]
            loc_act, loc_tot = torch.sum(loc_mask), loc_mask.numel()
            cls._stat_window_activation_nums[idx] += loc_act
            cls._stat_window_total_nums[idx] += loc_tot
        

    @classmethod
    def transfer_regularization(cls, raw_router_score: torch.Tensor, router_score: torch.Tensor,
                                segment_shift_indices: torch.Tensor, layer_id: int,
                                gate_policy: str, raw_router_gate: torch.Tensor = None):
        if len(cls._stat_act_window_sizes) == 0:
            return 0.
        assert raw_router_score.shape == router_score.shape, f"{raw_router_score.shape} != {router_score.shape}"
        seq_len, num_experts = router_score.shape

        # compute the transfer regularization loss
        valid_loss_mask = torch.ne(segment_shift_indices, -1)
        left_score = raw_router_score[segment_shift_indices][valid_loss_mask]
        right_score = raw_router_score[valid_loss_mask]
        left_gate, right_gate = None, None
        if raw_router_gate is not None:
            assert gate_policy == "fixdrelu"
            assert raw_router_score.shape == raw_router_gate.shape
            left_gate = raw_router_gate[segment_shift_indices][valid_loss_mask]
            right_gate = raw_router_gate[valid_loss_mask]

        accu_loss = 0.
        transfer_lambda = cls.get_current_transfer_lambda()
        if transfer_lambda > 0:
            if raw_router_gate is not None:
                # left_score_st, right_score_st = torch.sigmoid(left_score * cls._sigmoid_steep), torch.sigmoid(right_score * cls._sigmoid_steep)
                # left_gate_st, right_gate_st = torch.sigmoid(left_gate * cls._sigmoid_steep), torch.sigmoid(right_gate * cls._sigmoid_steep)
                # loss = F.binary_cross_entropy(
                #     left_score_st * left_gate_st, right_score_st * right_gate_st,
                # ) * transfer_lambda

                A, B, C, D = left_score * cls._sigmoid_steep, left_gate * cls._sigmoid_steep, right_score * cls._sigmoid_steep, right_gate * cls._sigmoid_steep
                loss = stable_bce(A, B, C, D) * transfer_lambda
            else:
                loss = F.binary_cross_entropy_with_logits(
                    left_score * cls._sigmoid_steep, torch.sigmoid(right_score * cls._sigmoid_steep),
                ) * transfer_lambda
            accu_loss = loss / cls._accu_iters
            cls._transfer_losses.append(loss.item())

        # stat token sharing activation rates
        if cls._inference_mode or router_score.requires_grad:
            cls.stat_window_activation(router_score, valid_loss_mask)

        if cls._inference_mode:
            with torch.no_grad():
                # stat expert activation
                act_mask = torch.ne(router_score[valid_loss_mask], 0)
                seq_len, expert_num = act_mask.shape
                assert len(cls._expert_activation_num) == len(cls._expert_total_num)
                if len(cls._expert_activation_num) <= layer_id:
                    assert len(cls._expert_activation_num) == layer_id, f"{len(cls._expert_activation_num)} | {layer_id}"
                    cls._expert_activation_num.append([0] * expert_num)
                    cls._expert_total_num.append([0] * expert_num)
                loc_exp_act = torch.sum(act_mask, dim=0).tolist()
                assert len(loc_exp_act) == expert_num
                for idx in range(expert_num):
                    cls._expert_activation_num[layer_id][idx] += loc_exp_act[idx]
                    cls._expert_total_num[layer_id][idx] += seq_len

        return accu_loss

    @classmethod
    def token_balance_regularization(
        cls, raw_router_score: torch.Tensor, segment_shift_indices: torch.Tensor, gate_policy: str,
    ):
        if len(cls._stat_act_window_sizes) == 0:
            return 0.
        assert gate_policy == "null"
        valid_loss_mask = torch.ne(segment_shift_indices, -1)
        left_score = raw_router_score[segment_shift_indices][valid_loss_mask]
        right_score = raw_router_score[valid_loss_mask]

        approx_left_act = torch.sigmoid(cls._sigmoid_steep * left_score).mean(dim=1)
        approx_right_act = torch.sigmoid(cls._sigmoid_steep * right_score).mean(dim=1)

        accu_loss = 0.
        if cls._token_balance_factor > 0:
            loss = F.mse_loss(approx_left_act, approx_right_act) * cls._token_balance_factor
            accu_loss = loss / cls._accu_iters
        return accu_loss

    @classmethod
    def get_clear_transfer_loss(cls):
        total_loss = 0.
        for loss in cls._transfer_losses:
            total_loss += loss
        cls._transfer_losses = []

        assert len(cls._stat_act_window_sizes) == len(cls._stat_window_activation_nums) == len(cls._stat_window_total_nums)
        window_sizes, activation_nums, total_nums = \
            cls._stat_act_window_sizes[:], cls._stat_window_activation_nums, cls._stat_window_total_nums
        cls._stat_window_activation_nums = [0] * len(cls._stat_act_window_sizes)
        cls._stat_window_total_nums = [0] * len(cls._stat_act_window_sizes)
        return total_loss, window_sizes, activation_nums, total_nums

    @classmethod
    def get_clear_expert_activations(cls):
        if len(cls._expert_activation_num) == 0 or len(cls._expert_activation_num[0]) == 0:
            return []
        layer_expert_act_rates = []
        for lid in range(len(cls._expert_activation_num)):
            loc_expert_act_rates = []
            for eid in range(len(cls._expert_activation_num[0])):
                loc_expert_act_rates.append(round(cls._expert_activation_num[lid][eid] * 100 / cls._expert_total_num[lid][eid], 2))
            layer_expert_act_rates.append(loc_expert_act_rates)
        cls._expert_activation_num, cls._expert_total_num = [], []
        return layer_expert_act_rates

    @classmethod
    def set_record_flag(cls, record_flag: bool):
        cls._record_flag = record_flag

    @classmethod
    def append_input(cls, inputs: torch.Tensor):
        if cls._record_flag:
            cls._input_list.append(inputs.clone().detach())

    @classmethod
    def append_output(cls, output: torch.Tensor):
        if cls._record_flag:
            cls._output_list.append(output.clone().detach())

    @classmethod
    def get_clear_records(cls) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        input_list, output_list = cls._input_list, cls._output_list
        cls._input_list, cls._output_list = [], []
        return input_list, output_list

    @classmethod
    def set_intermediate_flag(cls, intermediate_flag: bool):
        cls._intermediate_flag = intermediate_flag

    @classmethod
    def append_intermediate(cls, intermediate: torch.Tensor):
        if cls._intermediate_flag:
            cls._intermediate_list.append(intermediate.clone().detach())

    @classmethod
    def get_clear_intermediate(cls) -> List[torch.Tensor]:
        intermediate_list = cls._intermediate_list
        cls._intermediate_list = []
        return intermediate_list
