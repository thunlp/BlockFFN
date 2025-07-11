# coding=utf-8
# Copyright 2022 ModelBest Inc.

import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict

from transformers import LlamaTokenizerFast

import bmtrain as bmt
import torch

sys.path.append("../../")
from cpm.arguments import get_args
from cpm.dragonfly.modeling_dragonfly import Dragonfly
from cpm.dragonfly.modeling_dragonfly import DragonflyConfig
from cpm.dragonfly.training_tasks.pretrain_indexed import CudaPrefetcher
from cpm.dragonfly.training_tasks.pretrain_indexed import MixedIndexedDataset
from cpm.dragonfly.training_tasks.pretrain_indexed import UnpadBatchedMixedDataset
from cpm.utils import exporter
from cpm.utils import logger
from cpm.utils.exporter import save_every_step_stats
from cpm.utils.training_stats import num_non_embedding_parameters
from cpm.utils.training_stats import num_parameters
from cpm.dragonfly.activation_context import ActivationContext
from cpm.training_utils.value_scheduler import BaseValueScheduler
from cpm.training_utils.value_scheduler import AdaptiveLinearScheduler


def get_tokenizer(args):
    tokenizer = LlamaTokenizerFast(vocab_file=args.tokenizer_path)
    return tokenizer


def get_model(args):
    config = DragonflyConfig.from_json_file(args.model_config)
    config.tp = 1 if args.tp_size != 1 else 0  # TODO
    config.pose_prob = args.pose_prob
    config.pose_scaling_factor = args.pose_scaling_factor
    config.rope_scaling_type = args.rope_scaling_type
    config.rope_scaling_factor = args.rope_scaling_factor
    config.orig_max_length = args.orig_max_length
    if args.router_activate_fn:
        config.router_activate_fn = args.router_activate_fn
    if args.ffn_activate_fn:
        config.ffn_activate_fn = args.ffn_activate_fn
    if args.ffn_type:
        config.ffn_type = args.ffn_type
    config.norm_after_router = args.norm_after_router
    config.norm_scale = args.norm_scale
    config.use_checkpoint = True if args.use_checkpoint == 1 else False

    bmt.print_rank("model config: {}".format(config))
    bmt.print_rank("bmt config: {}".format(bmt.config))

    model = Dragonfly(config)
    if args.load is not None:
        bmt.print_rank("args.load is not None, start to load checkpoints:", args.load)
        exporter.load_model_ckpt(args, model)
    else:
        bmt.print_rank("args.load is None, start to initialize parameters")
        bmt.init_parameters(model)
    return model


def get_optimizer(args, model):
    if args.router_lr < 0:
        args.router_lr = args.lr
    scale_lr_group = []
    router_lr_group = []
    normal_group = []
    scale_lr_group_name, router_lr_group_name, normal_group_name = [], [], []
    for n, p in model.named_parameters():
        if "moe_router.w_gate" in n:
            router_lr_group.append(p)
            router_lr_group_name.append(n)
        elif (n.endswith(".weight") or "moe_router" in n or "moe_experts" in n) \
                and "layernorm" not in n and "embedding" not in n and "lm_head" not in n:
            scale_lr_group.append(p)
            scale_lr_group_name.append(n)
        else:
            normal_group.append(p)
            normal_group_name.append(n)
    bmt.print_rank(scale_lr_group_name, router_lr_group_name, normal_group_name)
    param_groups = [
        {"params": scale_lr_group, "lr": args.lr / model.config.scale_width},
        {"params": normal_group, "lr": args.lr},
    ]
    router_groups = [
        {"params": router_lr_group, "lr": args.router_lr / model.config.scale_width},
    ]

    if args.offload:
        optimizer = bmt.optim.AdamOffloadOptimizer(param_groups, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        router_optimizer = bmt.optim.AdamOffloadOptimizer(router_groups, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    else:
        optimizer = bmt.optim.AdamOptimizer(param_groups, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        router_optimizer = bmt.optim.AdamOptimizer(router_groups, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    if not args.force_restart and args.load is not None and args.load_grad:
        exporter.load_optimizer_ckpt(args, optimizer, is_router=False)
        exporter.load_optimizer_ckpt(args, router_optimizer, is_router=True)
        bmt.print_rank("optimizer is loaded!")
    return optimizer, router_optimizer


def get_learning_rate_scheduler(args, optimizer, factor: float, is_router: bool):
    from cpm.training_utils.lr_scheduler import Cosine
    from cpm.training_utils.lr_scheduler import WarmupStableDrop
    from cpm.training_utils.lr_scheduler import WarmupStableExp

    end_iter = args.train_iters
    if 0 < args.warmup_iters < 1:  # 需要支持按固定比例step用来做warmup的
        warmup_iters = int(end_iter * args.warmup_iters)
    else:
        warmup_iters = int(args.warmup_iters)

    if 0 < args.drop_iters < 1:  # 需要支持按固定比例step用来做drop的
        drop_iters = int(end_iter * args.drop_iters)
    else:
        drop_iters = int(args.drop_iters)
    start_step = args.start_step
    drop_begin = args.drop_begin

    end_iter, warmup_iters, start_step, drop_iters, drop_begin = int(end_iter * factor), int(warmup_iters * factor), \
        int(start_step * factor), int(drop_iters * factor), int(drop_begin * factor)
    lr = args.router_lr if is_router else args.lr

    if args.lr_scheduler == "cosine":
        lr_scheduler = Cosine(
            optimizer,
            start_lr=lr,
            warmup_iter=warmup_iters,
            end_iter=end_iter,  # 原来是lr_decay_iter
            num_iter=start_step,
            # lr_end_restart=args.lr_end_restart,
            resume_no_optimze=args.resume_no_optimze,
        )
    elif args.lr_scheduler == "warmupstabledrop":
        lr_scheduler = WarmupStableDrop(
            optimizer,
            start_lr=lr,
            warmup_iter=warmup_iters,
            end_iter=end_iter,  # 原来是lr_decay_iter
            drop_iter=drop_iters,
            num_iter=start_step,
            resume_no_optimze=args.resume_no_optimze,
        )
    elif args.lr_scheduler == "warmupstableexp":
        lr_scheduler = WarmupStableExp(
            optimizer,
            start_lr=lr,
            warmup_iter=warmup_iters,
            drop_begin=drop_begin,  # 原来是lr_decay_iter
            drop_iter=drop_iters,
            drop_rate=args.drop_rate,
            num_iter=start_step,
            resume_no_optimze=args.resume_no_optimze,
        )
    else:
        raise NotImplementedError(f"invalid lr_scheduler: {args.lr_scheduler}")
    return lr_scheduler


def get_value_scheduler(args):
    if not args.use_value_scheduler:
        return None
    if args.router_entropy_coef <= 0 and args.chunk_regularization_factor <= 0 and args.l1_lambda <= 0:
        return None
    accu_iters = args.grad_accum
    assert (args.router_entropy_coef > 0) + (args.chunk_regularization_factor > 0) + (args.l1_lambda > 0) == 1
    if args.router_entropy_coef > 0:
        print("Apply value scheduler for the router entropy regularization ......")
        start_val = args.router_entropy_coef
    elif args.chunk_regularization_factor > 0:
        print("Apply value scheduler for the chunk regularization ......")
        start_val = args.chunk_regularization_factor
    else:
        assert args.l1_lambda > 0
        print("Apply value scheduler for the l1 regularization ......")
        start_val = args.l1_lambda

    start_step = 0
    if not args.load_value_scheduler:
        start_step = args.start_step
    value_scheduler = AdaptiveLinearScheduler(
        start_val=start_val,
        warmup_iter=0, stable_iter=1000 * accu_iters, step_iter=100 * accu_iters,
        start_step=start_step * accu_iters,
        max_factor=args.scheduler_max_factor, min_factor=args.scheduler_min_factor,
    )
    if not args.force_restart and args.load is not None and args.load_value_scheduler:
        value_scheduler = exporter.load_value_scheduler(args, value_scheduler)
    return value_scheduler


def setup_model_and_optimizer(args):
    start = time.time()
    tokenizer = get_tokenizer(args)
    bmt.synchronize()
    logger.info("load tokenizer in {:.2f}s".format(time.time() - start))

    start = time.time()
    model = get_model(args)
    logger.info("load model in {:.2f}s".format(time.time() - start))

    start = time.time()
    optimizer, router_optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer, factor=1, is_router=False)
    router_accu_factor = args.grad_accum / args.router_accu_iters
    router_lr_scheduler = get_learning_rate_scheduler(args, router_optimizer, factor=router_accu_factor, is_router=True)
    bmt.synchronize()
    logger.info("load lr_scheduler in {:.2f}s".format(time.time() - start))

    start = time.time()
    value_scheduler = get_value_scheduler(args)
    bmt.synchronize()
    logger.info("load value_scheduler in {:.2f}s".format(time.time() - start))

    return tokenizer, model, optimizer, router_optimizer, lr_scheduler, router_lr_scheduler, value_scheduler


def initialize():
    args = get_args(pretrain=True, sparse_training=True)
    bmt.init_distributed(seed=args.seed, tp_size=args.tp_size)

    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
    if (not args.force_restart or args.valid_only) and args.load is not None:
        if args.only_load_model == 0:
            if args.start_step == 0:
                log_ckpt = exporter.load_log_ckpt(args)
                if "iteration" in log_ckpt:
                    former_accu = args.former_grad_accum if args.former_grad_accum > 0 else args.grad_accum
                    args.start_step = log_ckpt["iteration"] // former_accu
                else:
                    args.start_step = (int)(re.findall("(\d+)", args.load)[-1])
                logger.info("Start from step {}".format(args.start_step))
        elif args.only_load_model == 1:
            logger.info("You load model ckpt, and choose to completely start the 0 step.")
        else:
            raise NotImplementedError
    else:
        logger.info("You do not load model")
    args.force_restart = args.force_restart or args.valid_only

    return args


def see_memory(detail=False):
    if detail:
        res = torch.cuda.memory_summary()
    else:
        res = (
            round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024), 2),
            round(torch.cuda.memory_reserved() / (1024 * 1024 * 1024), 2),
            round(torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024), 2),
        )
    torch.cuda.reset_peak_memory_stats()
    return res


def add_mem_time(info, mem_usage, tim_usage):
    torch.cuda.synchronize()
    bmt.synchronize()
    mem_usage[info] = see_memory()
    tim_usage[info] = time.time()
    return mem_usage, tim_usage


def get_task_loss_and_token(loss, task_ids, task_num, targets):
    # task_ids 可能有-1 来代表无效token
    _task_num = task_num + 1
    _task_ids = (task_ids.clone() + 1).to(torch.int64)  # [batch_size, seq_len]
    # gen masks
    _task_mask = torch.zeros((_task_num, *_task_ids.shape), device=_task_ids.device)
    _task_mask.scatter_(0, _task_ids.unsqueeze(0), 1)  # [task_num, batch_size, seq_len]
    _loss_mask = torch.ne(targets, -100).to(torch.int32)
    _mask = _task_mask * _loss_mask.unsqueeze(0)  # [task_num, batch_size, seq_len]
    # calc loss and tokens
    _task_losses = (loss.unsqueeze(0) * _mask).view((_task_num, -1)).sum(dim=-1)[1:]  # [task_num]
    _task_tokens = _mask.view((_task_num, -1)).sum(dim=-1)[1:]  # [task_num]
    # return token-wise avg losses and tokens
    return torch.nan_to_num(_task_losses / _task_tokens, nan=0.0), _task_tokens


class ChunkAve:
    def __init__(self, chunk_size=100):
        self.ave_list = []
        self.chunk_size = chunk_size

    def record(self, time):
        self.ave_list.append(time)
        self.ave_list = self.ave_list[-self.chunk_size:]

    def get(self):
        return sum(self.ave_list) / len(self.ave_list)


def stat_intermediate(intermediate_list: list, model_unique: str):
    if bmt.rank() != 0:
        return
    global points, bins
    assert len(intermediate_list) > 0
    for intermediate in intermediate_list:
        for idx in range(len(bins)):
            mask = torch.ones(intermediate.shape, device=intermediate.device, dtype=torch.bool)
            if idx < len(points):
                mask = torch.logical_and(mask, torch.lt(intermediate, points[idx]))
            if idx > 0:
                mask = torch.logical_and(mask, torch.ge(intermediate, points[idx - 1]))
            bins[idx] += torch.sum(mask).item()
    os.makedirs("/data/checkpoints/stats", exist_ok=True)
    with open(f"/data/checkpoints/stats/bins_{model_unique}.json", "w") as fp:
        json.dump([points, bins], fp)


class GradRecorder:
    def __init__(self):
        self.grad = None

    def record(self, model):
        grad = self._record(model)
        self.grad = grad

    def _record(self, model):
        grad_list = []
        for n, p in model.named_parameters():
            if p.grad is not None:
                grad_list.append(p.grad)
        grad = torch.cat(grad_list)
        return grad

    def get_grad_second(self, model):
        new_grad = self._record(model)
        grad_diff = new_grad - self.grad
        grad_diff_abs = torch.abs(grad_diff).mean().item()
        grad_diff = grad_diff.mean().item()
        grad_dot = (new_grad * self.grad).sum().item()
        grad_norm_old = self.grad.norm().item()
        grad_norm_new = new_grad.norm().item()
        grad_max = new_grad.abs().max().item()
        self.grad = new_grad
        return {"grad_diff_abs": grad_diff_abs, "grad_diff": grad_diff, "grad_max": grad_max, "grad_dot": grad_dot,
                "grad_norm_old": grad_norm_old, "grad_norm_new": grad_norm_new}


@torch.inference_mode()
def do_evaluation(args, tokenizer, model: Dragonfly, iteration: int):
    mixed_validation_dataset = MixedIndexedDataset(
        cfg_path=args.valid_dataset,
        cfg_json_str=None,
        tokenizer=tokenizer,
        max_length=args.max_length,
        nthreads=args.dataloader_num_threads,
        prefetch_slice=args.dataloader_prefetch,
        weight_by_size=True,
        chunk_regularization_length=args.chunk_regularization_length,
    )
    mixed_validation_dataset = UnpadBatchedMixedDataset(mixed_validation_dataset, args.batch_size, args.max_length)
    validation_dataloader = torch.utils.data.DataLoader(
        mixed_validation_dataset,
        batch_size=None,
        collate_fn=lambda x: x,
        num_workers=args.dataloader_num_workers,
        prefetch_factor=args.dataloader_prefetch_factor,
    )

    ValidationDataIterator = CudaPrefetcher(validation_dataloader, tp_size=args.tp_size, tp_rank=bmt.config["tp_rank"])
    bmt.print_rank("Preparing validation dataset done.")

    model_config = model.config
    model.eval()
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100, reduction="none")

    global_total_task_token = defaultdict(int)  # token by task
    global_total_sum_loss = defaultdict(float)  # loss by task
    activation_rates = []
    mid_activation_rates = []
    layer_activation_rates = [[] for _ in range(model_config.num_layers)]
    stat_act_window_sizes = [int(s) for s in args.stat_act_window_sizes.split(",")]
    stat_act_window_rates = {f"transfer_rate_{s:02}": [] for s in stat_act_window_sizes}

    layer_expert_act_list = []

    last_data, task_names = None, None
    for valid_iteration, data in enumerate(ValidationDataIterator, start=1):
        if data is None:
            exhausted = torch.Tensor([1]).cuda()
            data = last_data
        else:
            exhausted = torch.Tensor([0]).cuda()
            last_data = data
        gathered_exhausted = bmt.distributed.all_gather(exhausted)
        # drop the last batch
        stop_valid_iteration = torch.sum(gathered_exhausted).item() > 0
        if stop_valid_iteration:
            break

        task_names = data["task_names"]
        logits, l1_loss, balance_loss, router_entropy, transfer_loss, chunk_loss, token_balance_loss = model(
            input=data["inputs"],
            cu_seqlens=data["cu_seqlens"],
            max_seqlen=data["max_seqlen"],
            position_ids=data["position_ids"],
            segment_shift_indices=data["segment_shift_indices"],
            seq_mask=data["seq_mask"],
        )
        if exhausted.item() == 1:
            continue

        # chunk targets and task_ids
        data["targets"] = (
            data["targets"]
            .view(-1)
            .chunk(bmt.config["tp_size"])[bmt.config["tp_rank"]]
            .view(data["targets"].shape[0], -1)
        )

        data["task_ids"] = (
            data["task_ids"]
            .view(-1)
            .chunk(bmt.config["tp_size"])[bmt.config["tp_rank"]]
            .view(data["task_ids"].shape[0], -1)
        )

        _target = data["targets"].view(-1)
        non_reduced_loss = loss_func(logits.view(-1, logits.size(-1)), _target)
        _w = (_target != -100).int()
        loss = non_reduced_loss.sum() / _w.sum().float()

        global_loss = bmt.distributed.all_reduce(loss, op="avg").item()

        _, activation_num, total_num, layer_activation_nums, layer_total_nums = ActivationContext.get_clear_loss(output_layer=True)
        if not isinstance(l1_loss, torch.Tensor):
            l1_loss = torch.tensor(l1_loss).to(loss)
        global_l1_loss = bmt.distributed.all_reduce(l1_loss, op="avg").item()
        activation_num, total_num = torch.tensor(activation_num, device=loss.device, dtype=torch.long), \
            torch.tensor(total_num, device=loss.device, dtype=torch.long)
        activation_num = bmt.distributed.all_reduce(activation_num, op="sum").item()
        total_num = bmt.distributed.all_reduce(total_num, op="sum").item()
        activation_rate = round(activation_num * 100 / total_num + 1e-10, 2) if total_num > 0 else 0.

        mid_activation_num, mid_total_num = ActivationContext.get_clear_moe_intermediate_activation()
        mid_activation_num, mid_total_num = torch.tensor(mid_activation_num, device=loss.device, dtype=torch.long), \
            torch.tensor(mid_total_num, device=loss.device, dtype=torch.long)
        mid_activation_num = bmt.distributed.all_reduce(mid_activation_num, op="sum").item()
        mid_total_num = bmt.distributed.all_reduce(mid_total_num, op="sum").item()
        mid_activation_rate = round(mid_activation_num * 100 / mid_total_num + 1e-10, 2) if mid_total_num > 0 else 0.
        mid_activation_rates.append(mid_activation_rate)

        assert len(layer_activation_nums) == len(layer_total_nums) == len(layer_activation_rates) == model_config.num_layers
        for lid, (layer_act, layer_tot, layer_ls) in enumerate(zip(layer_activation_nums, layer_total_nums, layer_activation_rates)):
            layer_act, layer_tot = torch.tensor(layer_act, device=loss.device, dtype=torch.long), torch.tensor(layer_tot, device=loss.device, dtype=torch.long)
            layer_act, layer_tot = bmt.distributed.all_reduce(layer_act, op="sum").item(), bmt.distributed.all_reduce(layer_tot, op="sum").item()
            layer_act_rate = round(layer_act * 100 / layer_tot + 1e-10, 2) if layer_tot > 0 else 0.
            layer_ls.append(layer_act_rate)

        _, trans_window_sizes, trans_activation_nums, trans_total_nums = ActivationContext.get_clear_transfer_loss()
        if not isinstance(transfer_loss, torch.Tensor):
            transfer_loss = torch.tensor(transfer_loss).to(loss)
        global_transfer_loss = bmt.distributed.all_reduce(transfer_loss, op="avg").item()
        assert len(trans_window_sizes) == len(trans_activation_nums) == len(trans_total_nums)
        transfer_activation_rates = {}
        for ws, act, tot in zip(trans_window_sizes, trans_activation_nums, trans_total_nums):
            act, tot = torch.tensor(act, device=loss.device, dtype=torch.long), torch.tensor(tot, device=loss.device, dtype=torch.long)
            act, tot = bmt.distributed.all_reduce(act, op="sum").item(), bmt.distributed.all_reduce(tot, op="sum").item()
            transfer_activation_rates[f"transfer_rate_{ws:02}"] = round(act * 100 / tot + 1e-10, 2) if tot > 0 else 0.
        transfer_activation_ls = sorted(list(transfer_activation_rates.items()), key=lambda x: x[0])

        if not isinstance(chunk_loss, torch.Tensor):
            chunk_loss = torch.tensor(chunk_loss).to(loss)
        global_chunk_loss = bmt.distributed.all_reduce(chunk_loss, op="avg").item()

        if not isinstance(token_balance_loss, torch.Tensor):
            token_balance_loss = torch.tensor(token_balance_loss).to(loss)
        global_token_balance_loss = bmt.distributed.all_reduce(token_balance_loss, op="avg").item()

        layer_expert_act_rates = ActivationContext.get_clear_expert_activations()
        if len(layer_expert_act_rates) > 0:
            layer_expert_act_list.append(torch.tensor(layer_expert_act_rates, dtype=torch.double))

        global_balance_loss = bmt.distributed.all_reduce(balance_loss, op="avg").item()
        global_router_entropy = bmt.distributed.all_reduce(router_entropy, op="avg").item()

        with torch.no_grad():
            task_num = len(data["task_names"])
            task_loss, task_token = get_task_loss_and_token(
                non_reduced_loss, data["task_ids"], task_num, data["targets"]
            )
            task_loss_map: Dict[str, torch.Tensor] = {}
            task_loss_token = task_loss * task_token
            for i in range(task_num):
                task_loss_map[data["task_names"][i]] = task_loss_token[i]  # .item()
                global_total_task_token[data["task_names"][i]] += task_token[i]  # .item()
                global_total_sum_loss[data["task_names"][i]] += task_loss_token[i]  # .item()

        bmt.print_rank(
            "=========================================" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        )
        bmt.print_rank((
            "| Iter: {iteration:6d} | loss: {loss:.4f} | l1_loss: {l1_loss:.4f} | transfer_loss: {transfer_loss:.4f} | chunk_loss: {chunk_loss:.4f} "
            "| token_balance_loss: {token_balance_loss:.4f} | balance_loss: {balance_loss:.4f} | router_entropy: {router_entropy:.4f} | act_rate: {act_rate:.2f} | mid_act_rate: {mid_act_rate:.2f} | "
            + " | ".join([f"{key}: {{{key}:.2f}}" for key, _ in transfer_activation_ls])
        ).format(
            iteration=valid_iteration,
            loss=global_loss,
            l1_loss=global_l1_loss,
            transfer_loss=global_transfer_loss,
            chunk_loss=global_chunk_loss,
            token_balance_loss=global_token_balance_loss,
            balance_loss=global_balance_loss,
            router_entropy = global_router_entropy,
            act_rate=activation_rate,
            mid_act_rate=mid_activation_rate,
            **transfer_activation_rates,
        ))

        activation_rates.append(activation_rate)
        for key, val in transfer_activation_rates.items():
            stat_act_window_rates[key].append(val)

        if 0 < args.valid_iters == valid_iteration:
            break

    assert task_names is not None
    for task_name in task_names:
        global_total_sum_loss[task_name] = sum(
            bmt.distributed.all_gather(global_total_sum_loss[task_name])).item()
        global_total_task_token[task_name] = sum(
            bmt.distributed.all_gather(global_total_task_token[task_name])).item()

    total_sum_loss = sum(global_total_sum_loss.values())
    total_tokens = sum(global_total_task_token.values())
    bmt.print_rank(
        f"[Valid at iter {iteration}] global_total_task_token: {global_total_task_token}, "
        f"global_total_sum_loss: {global_total_sum_loss}, total_sum_loss: {total_sum_loss}, "
        f"total_tokens: {total_tokens}")

    ave_activation_rate = round(sum(activation_rates) / len(activation_rates), 2)
    ave_mid_activation_rate = round(sum(mid_activation_rates) / len(mid_activation_rates), 2)
    ave_transfer_rates = {}
    for key, val in stat_act_window_rates.items():
        ave_transfer_rates[key] = round(sum(val) / len(val), 2)
    ave_layer_activation_rates = []
    for lid, layer_ls in enumerate(layer_activation_rates):
        ave_layer_activation_rates.append(round(sum(layer_ls) / len(layer_ls), 2))
    save_info_dict = {
        "iteration": iteration,
        "total_sum_loss": total_sum_loss,
        "total_tokens": total_tokens,
        "task_loss": global_total_sum_loss,
        "total_task_token": global_total_task_token,
        "ave_activation_rate": ave_activation_rate,
        "ave_mid_activation_rate": ave_mid_activation_rate,
        "ave_layer_activation_rates": ave_layer_activation_rates,
    }
    save_info_dict.update(ave_transfer_rates)
    assert global_total_sum_loss.keys() == global_total_task_token.keys()
    task_avg = {}
    for key in global_total_sum_loss.keys():
        task_avg[key] = global_total_sum_loss[key] / global_total_task_token[key]
    save_info_dict["task_avg_loss"] = task_avg
    save_every_step_stats(save_info_dict, args.save + "/validation")

    if len(layer_expert_act_list) > 0:
        ave_layer_expert_act = torch.mean(torch.stack(layer_expert_act_list), dim=0)
        torch.save(ave_layer_expert_act, args.save + "/validation/ave_layer_expert_act.pkl")

    bmt.print_rank(">" * 30, "validation result begin", "<" * 30)
    bmt.print_rank(save_info_dict)
    bmt.print_rank("*" * 10, "activation rate:", ave_activation_rate, "*" * 10)
    bmt.print_rank("*" * 10, "mid activation rate:", ave_mid_activation_rate, "*" * 10)
    for key, val in ave_transfer_rates.items():
        bmt.print_rank(f"{key}:", val)
    for lid in range(model_config.num_layers):
        bmt.print_rank("*" * 10, f"layer {lid:02} activation rate:", ave_layer_activation_rates[lid], "*" * 10)
    bmt.print_rank(">" * 30, "validation result end", "<" * 30)


def pretrain(
    args,
    tokenizer,
    model: Dragonfly,
    optimizer,
    router_optimizer,
    lr_scheduler: bmt.lr_scheduler.WarmupLRScheduler,
    router_lr_scheduler: bmt.lr_scheduler.WarmupLRScheduler,
    value_scheduler: AdaptiveLinearScheduler,
):
    if args.inference:
        torch.set_grad_enabled(False)
        model.eval()
        ActivationContext.set_inference_mode(True)
        ActivationContext.set_intermediate_flag(True)
    if args.valid_only:
        ActivationContext.set_inference_mode(True)
        ActivationContext.set_layer_stat_enabled(True)
    ave_model_time = ChunkAve(chunk_size=100)
    ave_iter_time = ChunkAve(chunk_size=100)
    accu_iters = args.grad_accum
    router_accu_iters = args.router_accu_iters

    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100, reduction="none")
    router_accu_factor = accu_iters / router_accu_iters
    optim_manager = bmt.optim.OptimManager(
        loss_scale=bmt.world_size(),
        loss_scale_steps=args.loss_scale_steps,
        loss_scale_factor=2,
        max_loss_scale=bmt.world_size(),
        min_loss_scale=bmt.world_size(),
    )
    router_optim_manager = bmt.optim.OptimManager(
        loss_scale=bmt.world_size(),
        loss_scale_steps=int(args.loss_scale_steps * router_accu_factor),
        loss_scale_factor=2,
        max_loss_scale=bmt.world_size(),
        min_loss_scale=bmt.world_size(),
    )
    optim_manager.add_optimizer(optimizer, lr_scheduler)
    router_optim_manager.add_optimizer(router_optimizer, router_lr_scheduler)
    ActivationContext.set_optim_manager(optim_manager)
    ActivationContext.set_l1_lambda_option(args.l1_lambda_option)
    ActivationContext.set_l1_lambda(args.l1_lambda)
    ActivationContext.set_end_l1_lambda(args.end_l1_lambda)
    ActivationContext.set_end_cosine_step(args.end_cosine_step)
    ActivationContext.set_accu_iters(accu_iters)
    ActivationContext.set_stat_act_window_sizes(args.stat_act_window_sizes)
    ActivationContext.set_transfer_lambda(args.transfer_lambda)
    ActivationContext.set_token_balance_factor(args.token_balance_factor)
    ActivationContext.set_sigmoid_steep(args.sigmoid_steep)
    ActivationContext.set_chunk_regularization_enabled(args.chunk_regularization_factor > 0)
    ActivationContext.set_chunk_regularization_length(args.chunk_regularization_length)
    with open(args.dataset) as fin:
        dataset_config = json.load(fin)
    task_num = len(dataset_config)
    if args.task_act_stat:
        ActivationContext.enable_task_act_stat(model.config.num_layers, task_num)

    start_step = args.start_step

    writer = None
    if args.tensorboard is not None and bmt.rank() == 0:
        import distutils.version  # noqa: F401

        from tensorboardX import SummaryWriter

        if not os.path.exists(args.tensorboard):
            os.makedirs(args.tensorboard)
        writer = SummaryWriter(log_dir=args.tensorboard)

    if not args.force_restart and args.load is not None:
        log_ckpt = exporter.load_log_ckpt(args)
    else:
        log_ckpt = {}
    global_token_pass = log_ckpt.get("global_token_pass", 0.0)
    global_total_task_token = defaultdict(int, log_ckpt.get("global_total_task_token", {}))  # token by task

    global_world_size = bmt.world_size()
    bmt.print_rank("Begin preparing dataset")
    if args.tp_size == 1 or bmt.config["tp_rank"] == 0:
        mixed_indexed_dataset = MixedIndexedDataset(
            cfg_path=args.dataset,
            cfg_json_str=None,
            tokenizer=tokenizer,
            max_length=args.max_length,
            nthreads=args.dataloader_num_threads,
            prefetch_slice=args.dataloader_prefetch,
            weight_by_size=True,
            chunk_regularization_length=args.chunk_regularization_length,
        )

        if not args.force_restart and args.load is not None and args.only_load_model == 0 and args.load_dataloader_ckpt == 1:
            exporter.load_dataloader_ckpt(args, mixed_indexed_dataset)

        batched_dataset = UnpadBatchedMixedDataset(mixed_indexed_dataset, args.batch_size, args.max_length)
        dataloader = torch.utils.data.DataLoader(
            batched_dataset,
            batch_size=None,
            collate_fn=lambda x: x,
            num_workers=args.dataloader_num_workers,
            prefetch_factor=args.dataloader_prefetch_factor,
        )
    else:
        def dummy_generator():
            while True:
                yield None

        mixed_indexed_dataset = dummy_generator()
        dataloader = mixed_indexed_dataset

    DataIterator = CudaPrefetcher(dataloader, tp_size=args.tp_size, tp_rank=bmt.config["tp_rank"])
    bmt.print_rank("Preparing dataset done.")

    # inspect at init
    model_inspect = bmt.inspect.inspect_model(model, "*")
    bmt.print_rank(bmt.inspect.format_summary(model_inspect))
    l1_lambda_cosine_iter = args.start_cosine_step

    assert (args.router_entropy_coef > 0) + (args.chunk_regularization_factor > 0) + (args.l1_lambda > 0) <= 1

    try:
        mem_usage, tim_usage = {}, {}
        mem_usage, tim_usage = add_mem_time("before_log", mem_usage, tim_usage)

        for iteration, data in enumerate(DataIterator, start=start_step * accu_iters + 1):
            if args.valid_only or (args.valid_interval > 0 and iteration % args.valid_interval == 0):
                do_evaluation(args, tokenizer, model, args.start_step)
                if args.valid_only:
                    exit()

            ActivationContext.set_cosine_step(l1_lambda_cosine_iter)
            if args.tp_size == 1 or bmt.config["tp_rank"] == 0:
                mixed_indexed_dataset.update_states(data["task_ids"], data["indexes"])

            mem_usage, tim_usage = add_mem_time("init", mem_usage, tim_usage)

            ActivationContext.set_stat_task_ids(data["task_ids"])
            logits, l1_loss, balance_loss, router_entropy, transfer_loss, chunk_loss, token_balance_loss = model(
                input=data["inputs"],
                cu_seqlens=data["cu_seqlens"],
                max_seqlen=data["max_seqlen"],
                position_ids=data["position_ids"],
                segment_shift_indices=data["segment_shift_indices"],
                seq_mask=data["seq_mask"],
            )
            ActivationContext.unset_stat_task_ids()

            if args.inference:
                intermediate_list = ActivationContext.get_clear_intermediate()
                stat_intermediate(intermediate_list, os.path.basename(args.save.strip("/")))
                continue

            # chunk targets and task_ids
            data["targets"] = (
                data["targets"]
                .view(-1)
                .chunk(bmt.config["tp_size"])[bmt.config["tp_rank"]]
                .view(data["targets"].shape[0], -1)
            )
            data["task_ids"] = (
                data["task_ids"]
                .view(-1)
                .chunk(bmt.config["tp_size"])[bmt.config["tp_rank"]]
                .view(data["task_ids"].shape[0], -1)
            )

            _target = data["targets"].view(-1)
            non_reduced_loss = loss_func(logits.view(-1, logits.size(-1)), _target)
            _w = (_target != -100).int()
            loss = non_reduced_loss.sum() / _w.sum().float()

            global_loss = bmt.distributed.all_reduce(loss, op="avg").item()
            loss = loss / accu_iters
            balance_loss = balance_loss / accu_iters
            router_entropy = router_entropy / accu_iters

            if not isinstance(balance_loss, torch.Tensor):
                balance_loss = torch.tensor(balance_loss).to(loss)
            global_balance_loss = bmt.distributed.all_reduce(balance_loss, op="avg").item()

            if not isinstance(router_entropy, torch.Tensor):
                router_entropy = torch.tensor(router_entropy).to(loss)
            global_router_entropy = bmt.distributed.all_reduce(router_entropy, op="avg").item()

            if not isinstance(chunk_loss, torch.Tensor):
                chunk_loss = torch.tensor(chunk_loss).to(loss)
            global_chunk_loss = bmt.distributed.all_reduce(chunk_loss, op="avg").item()

            if not isinstance(token_balance_loss, torch.Tensor):
                token_balance_loss = torch.tensor(token_balance_loss).to(loss)
            global_token_balance_loss = bmt.distributed.all_reduce(token_balance_loss, op="avg").item()

            if not isinstance(l1_loss, torch.Tensor):
                l1_loss = torch.tensor(l1_loss).to(loss)
            global_l1_loss = bmt.distributed.all_reduce(l1_loss, op="avg").item()

            cur_balance_loss_coef = args.balance_loss_coef if args.balance_loss_coef > 0 else 0.
            cur_router_entropy_coef = args.router_entropy_coef if args.router_entropy_coef > 0 else 0.
            cur_chunk_coef = args.chunk_regularization_factor if args.chunk_regularization_factor > 0 else 0.
            cur_l1_lambda = args.l1_lambda if args.l1_lambda > 0 else 0.
            if value_scheduler is not None:
                if cur_router_entropy_coef > 0:
                    cur_router_entropy_coef = value_scheduler.step(
                        iteration, global_router_entropy, can_step=(iteration % accu_iters == 0),
                    )
                elif cur_chunk_coef > 0:
                    cur_chunk_coef = value_scheduler.step(
                        iteration, global_chunk_loss, can_step=(iteration % accu_iters == 0),
                    )
                else:
                    assert cur_l1_lambda > 0
                    cur_l1_lambda = value_scheduler.step(
                        iteration, global_l1_loss, can_step=(iteration % accu_iters == 0),
                    )
            global_balance_loss *= cur_balance_loss_coef
            global_router_entropy *= cur_router_entropy_coef
            global_chunk_loss *= cur_chunk_coef
            global_l1_loss *= cur_l1_lambda

            loss = loss + l1_loss * cur_l1_lambda + balance_loss * cur_balance_loss_coef + \
                   router_entropy * cur_router_entropy_coef + transfer_loss + chunk_loss * cur_chunk_coef + token_balance_loss
            mem_usage, tim_usage = add_mem_time("forward", mem_usage, tim_usage)

            optim_manager.backward(loss)

            _, activation_num, total_num = ActivationContext.get_clear_loss()
            activation_num, total_num = torch.tensor(activation_num, device=loss.device, dtype=torch.long), \
                torch.tensor(total_num, device=loss.device, dtype=torch.long)
            activation_num = bmt.distributed.all_reduce(activation_num, op="sum").item()
            total_num = bmt.distributed.all_reduce(total_num, op="sum").item()
            activation_rate = round(activation_num * 100 / total_num + 1e-10, 2) if total_num > 0 else 0.

            mid_activation_num, mid_total_num = ActivationContext.get_clear_moe_intermediate_activation()
            mid_activation_num, mid_total_num = torch.tensor(mid_activation_num, device=loss.device, dtype=torch.long), \
                torch.tensor(mid_total_num, device=loss.device, dtype=torch.long)
            mid_activation_num = bmt.distributed.all_reduce(mid_activation_num, op="sum").item()
            mid_total_num = bmt.distributed.all_reduce(mid_total_num, op="sum").item()
            mid_activation_rate = round(mid_activation_num * 100 / mid_total_num + 1e-10, 2) if mid_total_num > 0 else 0.

            _, trans_window_sizes, trans_activation_nums, trans_total_nums = ActivationContext.get_clear_transfer_loss()
            if not isinstance(transfer_loss, torch.Tensor):
                transfer_loss = torch.tensor(transfer_loss).to(loss)
            global_transfer_loss = bmt.distributed.all_reduce(transfer_loss, op="avg").item()
            assert len(trans_window_sizes) == len(trans_activation_nums) == len(trans_total_nums)
            transfer_activation_rates = {}
            for ws, act, tot in zip(trans_window_sizes, trans_activation_nums, trans_total_nums):
                act, tot = torch.tensor(act, device=loss.device, dtype=torch.long), torch.tensor(tot, device=loss.device, dtype=torch.long)
                act, tot = bmt.distributed.all_reduce(act, op="sum").item(), bmt.distributed.all_reduce(tot, op="sum").item()
                transfer_activation_rates[f"transfer_rate_{ws:02}"] = round(act * 100 / tot + 1e-10, 2) if tot > 0 else 0.
            transfer_activation_ls = sorted(list(transfer_activation_rates.items()), key=lambda x: x[0])

            mem_usage, tim_usage = add_mem_time("backward", mem_usage, tim_usage)

            grad_accum_init_time = tim_usage["init"]

            if iteration % router_accu_iters == 0 or iteration == (args.train_iters * accu_iters):
                router_grad_norm = router_optim_manager.clip_grad_norm(router_optimizer.param_groups, args.clip_grad, norm_type=2)
                router_optim_manager.step()
                router_optim_manager.zero_grad()
            else:
                # dummy optim step
                router_grad_norm = torch.Tensor([0.0]).cuda()

            if iteration % accu_iters == 0 or iteration == (args.train_iters * accu_iters):
                grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, args.clip_grad, norm_type=2)
                optim_manager.step()
                if model.config.ffn_type == "adpblock":
                    BaseValueScheduler.step_all()
                optim_manager.zero_grad()
                mem_usage, tim_usage = add_mem_time("optim", mem_usage, tim_usage)
                model_time = tim_usage["optim"] - grad_accum_init_time
                ave_model_time.record(model_time)
                l1_lambda_cosine_iter += 1
            else:
                # dummy optim step
                grad_norm = torch.Tensor([0.0]).cuda()
                tim_usage["optim"] = tim_usage["backward"]
                mem_usage["optim"] = mem_usage["backward"]
                model_time = tim_usage["optim"] - grad_accum_init_time

            with torch.no_grad():
                task_num = len(data["task_names"])
                task_loss, task_token = get_task_loss_and_token(
                    non_reduced_loss, data["task_ids"], task_num, data["targets"]
                )
                task_loss_map: Dict[str, float] = {}
                gatherd_task_loss_map = bmt.distributed.all_gather(task_loss)
                gatherd_task_token_map = bmt.distributed.all_gather(task_token)
                gatherd_task_loss_token_map = gatherd_task_loss_map * gatherd_task_token_map
                sum_task_loss = gatherd_task_loss_token_map.sum(dim=0)
                tot_task_token = gatherd_task_token_map.sum(dim=0)
                ave_task_loss = sum_task_loss / tot_task_token
                for i in range(task_num):
                    task_loss_map[data["task_names"][i]] = ave_task_loss[i].item()
                    global_total_task_token[data["task_names"][i]] += tot_task_token[i].item()

            local_total_rate = torch.Tensor(
                [data["lengths"].float().mean() / (args.max_length * args.batch_size)]
            ).cuda()
            local_total_rate = bmt.distributed.all_reduce(local_total_rate, op="avg").item()
            global_token_pass += (
                (global_world_size // args.tp_size) * local_total_rate * args.max_length * args.batch_size
            )

            bmt.print_rank(
                "=========================================" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            )
            last_before_log_time = tim_usage["before_log"]
            mem_usage, tim_usage = add_mem_time("before_log", mem_usage, tim_usage)

            iter_time = tim_usage["before_log"] - last_before_log_time

            ave_iter_time.record(iter_time)

            # cur_l1_lambda = ActivationContext.get_current_l1_lambda()
            # cur_cosine_step = ActivationContext.get_cosine_step()
            train_info = {
                "time": iter_time,
                "iteration": iteration,
                "real_iteration": iteration // accu_iters,
                "loss": global_loss,
                "lr": lr_scheduler.current_lr,
                "r_lr": router_lr_scheduler.current_lr,
                "token_max": local_total_rate,
                "token_pass": global_token_pass,
                "throughout": args.max_length * args.batch_size * local_total_rate / ave_iter_time.get() / args.tp_size,
                "grad_norm": grad_norm.item(),
                "router_grad_norm": router_grad_norm.item(),
                "mask_max": ((data["targets"] >= 0).sum(-1).float().mean() / args.max_length).item(),
                "task_loss": task_loss_map,
                "total_task_token": global_total_task_token,
            }
            global_token_pass_str = convert_to_k_and_b(global_token_pass)

            time_report_str = "{model_time:.2f}={forward_time:.2f}+{backward_time:.2f}+{optim_time:.2f}".format(
                model_time=model_time,
                forward_time=tim_usage['forward']-tim_usage['init'],
                backward_time=tim_usage['backward']-tim_usage['forward'],
                optim_time=tim_usage['optim'] - tim_usage['backward']
            )
            bmt.print_rank(
                (
                    "| Iter: {iteration:6d} | Real_Iter: {real_iter:6d} | loss: {loss:.4f} | l1_lambda: {l1_lambda:.4f} | l1_loss: {l1_loss:.4f} | transfer_loss: {transfer_loss:.4f} "
                    + "| chunk_loss: {chunk_loss:.4f} | chunk_coef: {chunk_coef:.4f} | token_balance_loss: {token_balance_loss:.4f} "
                    + "| balance_loss: {balance_loss:.4f} | router_entropy: {router_entropy:.4f} | router_entropy_coef: {router_entropy_coef:.6f} | act_rate: {activation_rate:.2f} | mid_act_rate: {mid_activation_rate:.2f} "
                    + "| lr: {lr:.4e} | r_lr: {r_lr:.4e} | model_time: {model_time} | iter_time: {iter_time:.2f}| chunk_ave_time: {chunk_ave_time:.2f}"
                    + " token/max: {tokenrate:.4f} | mask/max: {maskrate:.4f} | grad_norm: {grad_norm:.4f} | router_grad_norm: {router_grad_norm:.4f} | global_token_pass (B):"
                    + "{global_token_pass} | mem_usage {mem_usage} | "
                    + " | ".join([f"{key}: {{{key}:.2f}}" for key, _ in transfer_activation_ls])
                ).format(
                    iteration=iteration,
                    real_iter=iteration // accu_iters,
                    loss=global_loss,
                    l1_lambda=cur_l1_lambda,
                    l1_loss=global_l1_loss,
                    transfer_loss=global_transfer_loss,
                    chunk_loss=global_chunk_loss,
                    chunk_coef=cur_chunk_coef,
                    token_balance_loss=global_token_balance_loss,
                    balance_loss=global_balance_loss,
                    router_entropy=global_router_entropy,
                    router_entropy_coef=cur_router_entropy_coef,
                    activation_rate=activation_rate,
                    mid_activation_rate=mid_activation_rate,
                    lr=lr_scheduler.current_lr,
                    r_lr=router_lr_scheduler.current_lr,
                    model_time=time_report_str,
                    iter_time=iter_time,
                    chunk_ave_time=ave_iter_time.get(),
                    tokenrate=data["lengths"].float().mean() / args.max_length / args.batch_size,
                    maskrate=(data["targets"] >= 0).sum(-1).float().mean() / args.max_length / args.batch_size,
                    grad_norm=grad_norm.item(),
                    router_grad_norm=router_grad_norm.item(),
                    global_token_pass=global_token_pass_str,
                    mem_usage=max([value for key, value in mem_usage.items()]),
                    # cosine_iter=cur_cosine_step,
                    **transfer_activation_rates,
                )
            )

            bmt.print_rank(
                "task_loss:\t| "
                + " | ".join(["{}: {:.4f}".format(task_name, loss) for task_name, loss in task_loss_map.items()])
                + " |"
            )

            if iteration % 10 == 0:
                bmt.print_rank(
                    "task_tokens (B):\t| "
                    + " | ".join(
                        [
                            "{}: {:.4f}".format(task_name, task_token / 10**9)
                            for task_name, task_token in global_total_task_token.items()
                        ]
                    )
                    + " |"
                )

            if args.task_act_stat:
                task_names = data["task_names"]
                _, task_activation_rates = ActivationContext.get_task_act_stat()
                task_activation_rates = torch.mean(task_activation_rates, dim=0)
                task_activation_rates = bmt.distributed.all_reduce(task_activation_rates, op="avg")
                task_name_to_activation = {}
                for i in range(task_num):
                    task_name_to_activation[task_names[i]] = task_activation_rates[i].item()
                bmt.print_rank(
                    "task_act_rate (%):\t| "
                    + " | ".join(
                        [
                            "{}: {:.4f}".format(task_name, round(act_rate * 100, 2))
                            for task_name, act_rate in task_name_to_activation.items()
                        ]
                    )
                    + " |"
                )
            bmt.print_rank(f"*** activation: {activation_num}; total: {total_num} ***")
            bmt.print_rank(f"*** activation_rate: {activation_rate} ***")

            if iteration % (args.inspect_iters * accu_iters) == 0:
                model_inspect = bmt.inspect.inspect_model(model, "*")
                bmt.print_rank(bmt.inspect.format_summary(model_inspect))

            train_info["l1_lambda"] = cur_l1_lambda
            train_info["l1_loss"] = global_l1_loss
            train_info["balance_loss"] = global_balance_loss
            train_info["router_entropy"] = global_router_entropy
            train_info["router_entropy_coef"] = cur_router_entropy_coef
            train_info["activation_rate"] = activation_rate
            # train_info["cosine_iter"] = cur_cosine_step
            train_info["l1_lambda"] = cur_l1_lambda
            train_info["transfer_loss"] = global_transfer_loss
            train_info["chunk_loss"] = global_chunk_loss
            train_info["chunk_coef"] = cur_chunk_coef
            train_info["token_balance_loss"] = global_token_balance_loss
            train_info.update(transfer_activation_rates)
            if args.log_dir is not None and bmt.rank() == 0:
                if args.save is not None:
                    save_every_step_stats(train_info, args.save)

            if args.tensorboard is not None and bmt.rank() == 0:
                writer.add_scalar("Loss/train", global_loss, iteration)
                writer.add_scalar("Loss/train/l1_lambda", cur_l1_lambda, iteration)
                writer.add_scalar("Loss/train/l1_reg", global_l1_loss, iteration)
                writer.add_scalar("Loss/train/balance", global_balance_loss, iteration)
                writer.add_scalar("Loss/train/entropy", global_router_entropy, iteration)
                writer.add_scalar("Loss/train/entropy_coef", cur_router_entropy_coef, iteration)
                writer.add_scalar("Loss/train/transfer_loss", global_transfer_loss, iteration)
                writer.add_scalar("Loss/train/chunk_loss", global_chunk_loss, iteration)
                writer.add_scalar("Loss/train/chunk_coef", cur_chunk_coef, iteration)
                writer.add_scalar("Loss/train/token_balance_loss", global_token_balance_loss, iteration)
                writer.add_scalar("Act/train/act_rate", activation_rate, iteration)
                writer.add_scalar("Act/train/l1_lambda", cur_l1_lambda, iteration)
                writer.add_scalar("Optimizer/lr", lr_scheduler.current_lr, iteration)
                writer.add_scalar("Optimizer/scale", optim_manager.loss_scale, iteration)
                writer.add_scalar("Optimizer/grad_norm", grad_norm.item(), iteration)
                writer.add_scalar("R_Optimizer/lr", router_lr_scheduler.current_lr, iteration)
                writer.add_scalar("R_Optimizer/scale", router_optim_manager.loss_scale, iteration)
                writer.add_scalar("R_Optimizer/grad_norm", router_grad_norm.item(), iteration)
                for task_name, loss in task_loss_map.items():
                    if not math.isnan(loss):
                        writer.add_scalar("Loss/train/{}".format(task_name), loss, iteration)
                for key, val in transfer_activation_rates.items():
                    writer.add_scalar(f"Act/train/{key}", val, iteration)

            # -------- save file. If need to backup by Klara platform, use export.xx_save --------
            log_ckpt = {
                "global_total_task_token": global_total_task_token,
                "global_token_pass": global_token_pass,
                "iteration": iteration,
            }

            if args.save is not None and iteration % (args.save_iters * accu_iters) == 0:
                real_iter = iteration // accu_iters
                exporter.export(
                    model,
                    mixed_indexed_dataset,
                    tokenizer,
                    optimizer,
                    router_optimizer,
                    value_scheduler,
                    real_iter,
                    args,
                    log_ckpt=log_ckpt,
                    final_save=False,
                    async_save=args.async_save,
                )

            if iteration == (args.train_iters * accu_iters) and args.stop_when_end == 1:
                break

    except Exception as e:
        print(f"train loop err: {e}")
        raise e
    finally:
        pass

    if not args.inference and args.lr > 0:
        exporter.export(model, mixed_indexed_dataset, tokenizer, optimizer, router_optimizer, value_scheduler, -1, args, final_save=False)


def convert_to_k_and_b(number):
    if number >= 1e9:  # 大于或等于10亿
        b_number = number / 1e9
        return f"{b_number:.2f}B"
    elif number >= 1e6:  # 大于或等于1百万
        k_number = number / 1e6
        return f"{k_number:.2f}M"
    elif number >= 1e3:
        k_number = number / 1e3
        return f"{k_number:.2f}K"
    else:
        return str(number)


def main():
    args = initialize()
    bmt.synchronize()
    tokenizer, model, optimizer, router_optimizer, lr_scheduler, router_lr_scheduler, value_scheduler = setup_model_and_optimizer(args)
    bmt.print_rank("finish loading")
    bmt.print_rank(
        "Number of parameter {}, Number of non-e parameter {}".format(
            num_parameters(model), num_non_embedding_parameters(model)
        )
    )
    bmt.print_rank("args: {}".format(args))
    pretrain(args, tokenizer, model, optimizer, router_optimizer, lr_scheduler, router_lr_scheduler, value_scheduler)


if __name__ == "__main__":
    main()
