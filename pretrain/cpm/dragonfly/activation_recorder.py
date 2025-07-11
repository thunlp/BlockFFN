import os
import torch
import numpy as np
import bmtrain as bmt
from typing import List, Optional


class ActivationRecorder:

    record_flag: bool = False
    attn_flag: bool = False
    entry_num: int = 400000 // bmt.world_size()
    mlp_remain_ratio: float = -1
    record_path: str = "/data/checkpoints/activation_records/sft_lambda_1e2_643803_420000_job_646998_ckpt_6000"
    tmp_token_mask: Optional[torch.Tensor] = None
    activation_recorders: List = []

    @classmethod
    def set_record_flag(cls, flag: bool):
        cls.record_flag = flag

    @classmethod
    def set_attn_flag(cls, flag: bool):
        cls.attn_flag = flag

    @classmethod
    def set_entry_num(cls, entry_num: int):
        cls.entry_num = entry_num // bmt.world_size()

    @classmethod
    def set_mlp_remain_ratio(cls, mlp_remain_ratio: float):
        cls.mlp_remain_ratio = mlp_remain_ratio

    @classmethod
    def set_record_path(cls, path: str):
        os.makedirs(path, exist_ok=True)
        cls.record_path = path

    @classmethod
    def set_tmp_token_mask(cls, tmp_token_mask: torch.Tensor):
        cls.tmp_token_mask = tmp_token_mask

    @classmethod
    def unset_tmp_token_mask(cls):
        cls.tmp_token_mask = None

    def __init__(self, dim_model: int, dim_ff: int, num_heads: int, layer_idx: int):
        self.layer_idx = layer_idx
        self.dim_model = dim_model
        self.dim_ff = dim_ff
        self.num_heads = num_heads
        if not self.record_flag:
            return

        self.fp_mlp_query = np.memmap(
            f"{self.record_path}/mlp_query_rank{bmt.rank():02}_{layer_idx}.mmap",
            dtype="float16",
            mode="w+",
            shape=(self.entry_num, dim_model),
        )
        self.fp_mlp_label = np.memmap(
            f"{self.record_path}/mlp_label_rank{bmt.rank():02}_{layer_idx}.mmap",
            dtype="float16",
            mode="w+",
            shape=(self.entry_num, dim_ff),
        )
        self.fp_mlp_id = 0

        if self.attn_flag:
            self.fp_attn_query = np.memmap(
                f"{self.record_path}/attn_query_rank{bmt.rank():02}_{layer_idx}.mmap",
                dtype="float16",
                mode="w+",
                shape=(self.entry_num, dim_model),
            )
            self.fp_attn_label = np.memmap(
                f"{self.record_path}/attn_label_rank{bmt.rank():02}_{layer_idx}.mmap",
                dtype="float16",
                mode="w+",
                shape=(self.entry_num, num_heads),
            )
        self.fp_attn_id = 0

    @classmethod
    def add_recorder(cls, dim_model: int, num_expert: int, num_heads: int, layer_id: int):
        if not cls.record_flag:
            return
        cls.activation_recorders.append(ActivationRecorder(dim_model, num_expert, num_heads, layer_id))

    @classmethod
    def get_recorder(cls, layer_id: int):
        return cls.activation_recorders[layer_id]

    @classmethod
    def record_fp_mlp_query(cls, layer_id: int, hidden_states: torch.Tensor):
        if not cls.record_flag:
            return
        self: ActivationRecorder = cls.activation_recorders[layer_id]
        mask_1d = self.tmp_token_mask
        batch_size, seq_len = hidden_states.shape[:2]
        assert batch_size == mask_1d.shape[0]
        assert seq_len == mask_1d.shape[1]
        assert hidden_states.size(-1) == self.dim_model
        hidden_states = hidden_states.half()
        if self.fp_mlp_id < self.fp_mlp_query.shape[0]:
            _hidden_states = hidden_states.view(-1, self.dim_model)[mask_1d.bool().view(-1)]
            begin, end = self.fp_mlp_id, min(
                self.fp_mlp_id + _hidden_states.size(0), self.fp_mlp_query.shape[0]
            )
            self.fp_mlp_query[begin:end] = (
                _hidden_states[: end - begin].detach().cpu().numpy()
            )

    @classmethod
    def record_fp_mlp_label(cls, layer_id: int, up_proj: torch.Tensor):
        if not cls.record_flag:
            return
        self: ActivationRecorder = cls.activation_recorders[layer_id]
        mask_1d = self.tmp_token_mask
        batch_size, seq_len = up_proj.shape[:2]
        assert batch_size == mask_1d.shape[0]
        assert seq_len == mask_1d.shape[1]
        assert up_proj.size(-1) == self.dim_ff
        up_proj = torch.abs(up_proj.half())
        if self.mlp_remain_ratio > 0:
            remain_num = int(self.mlp_remain_ratio * up_proj.shape[-1])
            values, indices = torch.topk(up_proj, k=remain_num, dim=-1)
            new_up_proj = torch.zeros_like(up_proj)
            new_up_proj.scatter_(2, indices, values)
            up_proj = new_up_proj
        if self.fp_mlp_id < self.fp_mlp_label.shape[0]:
            label = up_proj.view(-1, self.dim_ff)[mask_1d.bool().view(-1)]
            begin, end = self.fp_mlp_id, min(
                self.fp_mlp_id + label.size(0), self.fp_mlp_label.shape[0]
            )
            self.fp_mlp_label[begin:end] = label[:end - begin].detach().cpu().numpy()
            self.fp_mlp_id += label.size(0)

    @classmethod
    def record_fp_attn_query(cls, layer_id: int, hidden_states: torch.Tensor):
        if not cls.record_flag or not cls.attn_flag:
            return
        self: ActivationRecorder = cls.activation_recorders[layer_id]
        mask_1d = self.tmp_token_mask
        batch_size, seq_len = hidden_states.shape[:2]
        assert batch_size == mask_1d.shape[0]
        assert seq_len == mask_1d.shape[1]
        assert hidden_states.size(-1) == self.dim_model
        hidden_states = hidden_states.half()
        if self.fp_attn_id < self.fp_attn_query.shape[0]:
            _hidden_states = hidden_states.view(-1, self.dim_model)[mask_1d.bool().view(-1)]
            begin, end = self.fp_attn_id, min(
                self.fp_attn_id + _hidden_states.size(0), self.fp_attn_query.shape[0]
            )
            self.fp_attn_query[begin:end] = (
                _hidden_states[: end - begin].detach().cpu().numpy()
            )

    @classmethod
    def record_fp_attn_label(cls, layer_id: int, attn_output: torch.Tensor):
        if not cls.record_flag or not cls.attn_flag:
            return
        self: ActivationRecorder = cls.activation_recorders[layer_id]
        mask_1d = self.tmp_token_mask
        batch_size, seq_len = attn_output.shape[:2]
        assert batch_size == mask_1d.shape[0]
        assert seq_len == mask_1d.shape[1]
        # attn_output: batch_size, len_q, num_heads, dim_head
        attn_output = attn_output.half()
        if self.fp_attn_id < self.fp_attn_label.shape[0]:
            attn_output_norm = attn_output.norm(dim=-1)
            assert attn_output_norm.size(-1) == self.num_heads
            attn_output_norm = attn_output_norm.view(-1, self.num_heads)[mask_1d.bool().view(-1)]

            begin, end = self.fp_attn_id, min(
                self.fp_attn_id + attn_output_norm.size(0), self.fp_attn_label.shape[0]
            )
            self.fp_attn_label[begin:end] = (
                attn_output_norm[: end - begin].detach().cpu().numpy()
            )
            self.fp_attn_id += attn_output_norm.size(0)

    @classmethod
    def check_id_ok(cls, layer_id: int):
        if not cls.record_flag:
            return
        self = cls.activation_recorders[layer_id]
        assert not self.attn_flag or self.fp_mlp_id == self.fp_attn_id, f"{self.fp_mlp_id} | {self.fp_attn_id}"


class ActivationCollector:
    _cur_layer_id = 0
    _total_layer_num = -1
    _dim_model = -1
    _dim_ff = -1
    _add_gate_score = False
    _sample_num_per_layer = 100000
    _output_dir = ""
    _arrays: List[np.ndarray] = []
    _counters: List[int] = []
    _pos_counters: List[int] = []
    _neg_counters: List[int] = []
    _matrix_adj: Optional[torch.Tensor] = None
    _save_activations = False
    _calculate_adj = False
    _meta: Optional[dict] = None

    def __new__(cls, *args, **kwargs):
        raise RuntimeError("ActivationCollector object is not allowed to be created!")

    @classmethod
    def is_enabled(cls):
        return cls._total_layer_num > 0

    @classmethod
    def enable_and_initialize(cls, total_layer_num: int, dim_model: int, dim_ff: int, add_gate_score: bool,
                              sample_num_per_layer: int, output_dir: str,
                              save_activations: bool = False, calculate_adj: bool = False, meta: dict = None):
        cls.disable()
        cls._total_layer_num = total_layer_num
        cls._dim_model = dim_model
        cls._dim_ff = dim_ff
        cls._add_gate_score = add_gate_score
        cls._sample_num_per_layer = sample_num_per_layer
        cls._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        cls._save_activations = save_activations
        cls._calculate_adj = calculate_adj
        cls._meta = meta
        target_dim = dim_model + dim_ff
        if cls._add_gate_score:
            target_dim += dim_ff
        for layer_id in range(total_layer_num):
            if save_activations:
                arr = np.memmap(
                    os.path.join(cls._output_dir, f"mlp_rank{bmt.rank():02}_layer{layer_id:02}.npy"), mode="w+",
                    dtype="float16", shape=(sample_num_per_layer, target_dim),
                )
                cls._arrays.append(arr)
            cls._counters.append(0)
            cls._pos_counters.append(0)
            cls._neg_counters.append(0)
        if calculate_adj:
            cls._matrix_adj = torch.zeros(total_layer_num, dim_ff, dim_ff, dtype=torch.float32, device="cuda")

    @classmethod
    def disable(cls):
        cls._total_layer_num, cls._cur_layer_id, cls._save_activations = -1, 0, False
        cls._arrays = []
        cls._counters = []

    @classmethod
    def record_activations(cls, inputs, act, gate_score=None):
        if not cls.is_enabled():
            return
        assert inputs.shape[-1] == cls._dim_model and act.shape[-1] == cls._dim_ff
        inputs, act = inputs.view(-1, cls._dim_model), act.view(-1, cls._dim_ff)
        ori_batch_size = inputs.shape[0]
        assert act.shape[0] == ori_batch_size

        layer_cnt = cls._counters[cls._cur_layer_id]
        batch_size = ori_batch_size
        if layer_cnt + batch_size > cls._sample_num_per_layer:
            batch_size = cls._sample_num_per_layer - layer_cnt
        # assert batch_size > 0

        if cls._save_activations:
            inputs_np, act_np = inputs.detach().cpu().half().numpy(), act.detach().cpu().half().numpy()
            cur_row = np.concatenate((inputs_np[:batch_size], act_np[:batch_size]), axis=-1)

            if cls._add_gate_score:
                assert gate_score is not None and gate_score.shape[-1] == cls._dim_ff
                gate_score = gate_score.view(-1, cls._dim_ff)
                assert gate_score.shape[0] == ori_batch_size
                gate_score = gate_score.detach().cpu().half().numpy()
                cur_row = np.concatenate((cur_row, gate_score[:batch_size]), axis=-1)
            cls._arrays[cls._cur_layer_id][layer_cnt:layer_cnt+batch_size] = cur_row
        if cls._calculate_adj:
            cls.update_matrix_adj(torch.abs(act[:batch_size]))

        cls._counters[cls._cur_layer_id] = layer_cnt + batch_size
        cls._pos_counters[cls._cur_layer_id] += torch.sum(act > 0).item()
        cls._neg_counters[cls._cur_layer_id] += torch.sum(act <= 0).item()
        cls._cur_layer_id = (cls._cur_layer_id + 1) % cls._total_layer_num

    @classmethod
    def update_matrix_adj(cls, activations: torch.Tensor):
        layer_id = cls._cur_layer_id
        bmt.print_rank(f"updating matrix_adj at layer #{layer_id:02}")
        overall_scaling = cls._meta["overall_scaling"]
        sub_batch_size = cls._meta["sub_batch_size"]
        upper_bound = cls._meta["upper_bound"]
        activations = activations.float() * overall_scaling
        assert torch.sum(activations < 0) == 0
        res = activations.unsqueeze(-1)  # [seq_length, dim_adj, 1]
        total_batch_size = res.shape[0]
        sub_batch_number = (total_batch_size + sub_batch_size - 1) // sub_batch_size
        tmp_counter = cls._counters[layer_id]
        for bid in range(sub_batch_number):
            batch_res = res[bid * sub_batch_size:(bid + 1) * sub_batch_size]
            tmp_batch_size = batch_res.shape[0]
            batch_res = torch.clamp(torch.bmm(batch_res, batch_res.transpose(1, 2)).mean(0), max=upper_bound)
            next_adj = cls._matrix_adj[layer_id] + (batch_res - cls._matrix_adj[layer_id]) * (
                        tmp_batch_size / (tmp_counter + tmp_batch_size))
            if torch.sum(torch.isnan(next_adj)) + torch.sum(torch.isinf(next_adj)) > 0:
                print(f"Rank {bmt.rank()} WARNING: nan/inf detected!!!")
                continue
            cls._matrix_adj[layer_id] = next_adj
            tmp_counter += tmp_batch_size

    @classmethod
    def check_full(cls):
        assert cls._cur_layer_id == 0
        ref = cls._counters[0]
        assert all(cnt == ref for cnt in cls._counters), str(cls._counters)
        return ref >= cls._sample_num_per_layer

    @classmethod
    def get_sample_num(cls):
        return cls._counters[0]

    @classmethod
    def get_counters(cls):
        return cls._counters[:], cls._pos_counters[:], cls._neg_counters[:]

    @classmethod
    def get_matrix_adj(cls):
        return cls._matrix_adj
