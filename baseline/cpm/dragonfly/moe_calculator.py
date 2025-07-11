'''
Author: yanhui@modelbest.cn
Date: 2024-02-18 21:04:29
LastEditors: yanhui@modelbest.cn
LastEditTime: 2024-02-19 16:47:50
FilePath: /CPM-Live-Luca/cpm/dragonfly/moe_calculator.py
Description: 

Copyright (c) 2024 by ModelBest, All Rights Reserved. 
'''
import tree
import torch
import torch.nn.functional as F
from fmoe.layers import _fmoe_general_global_forward


class TopKCalculator:
    def __init__(self, num_experts):
        self.num_experts = num_experts

    def forward(self, hidden_states, topk_indices, topk_weights, experts, **kwargs):
        orig_shape = hidden_states.shape
        top_k = topk_indices.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        # [bs * seq_len, dim_model]
        topk_idx_flat = topk_indices.view(-1)
        # [bs * seq_len * topk]
        # if self.training:
        # [bs * seq_len * topk, dim_model]
        hidden_states = hidden_states.repeat_interleave(top_k, dim=0)
        y = torch.empty_like(hidden_states)
        for i in range(self.num_experts):
            y[topk_idx_flat == i] = experts[i](hidden_states[topk_idx_flat == i])["hidden_states"]
        # [bs * seq_len, topk, dim_model], [bs * seq_len, topk, -1] -> [bs * seq_len, topk, dim_model] -> [bs * seq_len, dim_model]
        y = (y.view(*topk_weights.shape, -1) * topk_weights.unsqueeze(-1)).sum(dim=1)
        # [bs, seq_len, dim_model]
        y = y.view(*orig_shape)
        return y


class FastTopKCalculator:
    def __init__(self, num_experts):
        self.num_experts = num_experts

    def forward(self, hidden_states, topk_indices, topk_weights, experts, **kwargs):
        batch_size, seq_len, dim = hidden_states.shape
        top_k = topk_indices.shape[-1]
        hidden_states = hidden_states.view(batch_size * seq_len, dim)
        try:
            fwd = _fmoe_general_global_forward(
                hidden_states, topk_indices, experts, self.num_experts, world_size=1, experts=experts,
            )
        except RuntimeError as e:
            import bmtrain as bmt
            torch.save((hidden_states, topk_indices, experts, self.num_experts), f"../../debug/err_{bmt.rank()}.pkl")
            print(f"Rank {bmt.rank()} saved to:", f"../../debug/err_{bmt.rank()}.pkl")
            raise e

        def view_func(tensor):
            n_dim = tensor.shape[-1]
            tensor = tensor.view(-1, top_k, n_dim)
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


class ExpertChoiceCalculator:
    def __init__(self, num_experts):
        self.num_experts = num_experts

    def forward(self, hidden_states, topk_indices, topk_weights, experts, **kwargs):
        orig_shape = hidden_states.shape
        # [bs, seq_len, dim_model] -> [bs * seq_len, dim_model]
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        # [num_experts, top_k] -> [num_experts, top_k, bs * seq_len]
        topk_indices_onehot = F.one_hot(topk_indices, num_classes=hidden_states.shape[0])
        # get expert input
        expert_input = torch.einsum("ekl,ld->ekd", topk_indices_onehot.to(hidden_states.dtype), hidden_states)
        expert_output = torch.empty_like(expert_input)
        for i in range(self.num_experts):
            expert_output[i] = experts[i](expert_input[i])["hidden_states"]
        # aggregate reprs of different experts
        # [num_experts * top_k, bs * seq_len] -> [bs * seq_len, num_experts * top_k]
        P = topk_indices_onehot.view(-1, topk_indices_onehot.shape[-1]).transpose(0, 1)
        # [num_experts, top_k] -> [num_experts * top_k]
        topk_weights = topk_weights.view(-1)
        all_weights = P * topk_weights

        expert_output = expert_output.view(-1, expert_output.shape[-1])
        y = torch.matmul(all_weights, expert_output)
        # z=torch.einsum("ekl,ek,ekd->ld",topk_indices_onehot,weights,expert_output)
        y =  y.view(*orig_shape)
        return y
