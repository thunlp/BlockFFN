# coding=utf-8
# Copyright 2022 ModelBest Inc.

import math
import os
import re
import sys
import time
from collections import defaultdict
from itertools import chain
from typing import Dict

import bmtrain as bmt
import torch

sys.path.append("../../")
from cpm.arguments import get_args
from cpm.dragonfly.modeling_dragonfly import Dragonfly
from cpm.dragonfly.modeling_dragonfly import DragonflyConfig
from cpm.dragonfly.activation import ActivationContext
from cpm.dragonfly.training_tasks.pretrain_indexed import CudaPrefetcher
from cpm.dragonfly.training_tasks.pretrain_indexed import MixedIndexedDataset
from cpm.dragonfly.training_tasks.pretrain_indexed import UnpadBatchedMixedDataset
from cpm.utils import exporter
from cpm.utils import logger
from cpm.utils.exporter import save_every_step_stats
from cpm.utils.training_stats import num_non_embedding_parameters
from cpm.utils.training_stats import num_parameters


def get_tokenizer(args):
    from transformers import LlamaTokenizerFast

    if args.is_pretrained_tokenizer:
        tokenizer = LlamaTokenizerFast.from_pretrained(args.tokenizer_path)
    else:
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
    if args.moe_top_k > 0:
        config.moe_top_k = args.moe_top_k
    if args.moe_top_p > 0:
        config.moe_top_p = args.moe_top_p
    if args.moe_routing_strategy != "":
        config.moe_routing_strategy = args.moe_routing_strategy
    if args.num_shared_experts > 0:
        config.num_shared_experts = args.num_shared_experts
    if args.activate_fn != "":
        config.activate_fn = args.activate_fn

    bmt.print_rank("model config: {}".format(config))
    bmt.print_rank("bmt config: {}".format(bmt.config))

    model = Dragonfly(config)
    if args.load is not None:
        bmt.print_rank("args.load is not None, start to load checkpoints" + args.load)
        bmt.init_parameters(model)
        exporter.load_model_ckpt(args, model)
    else:
        bmt.print_rank("args.load is None, start to initialize parameters")
        bmt.init_parameters(model)
    return model


def get_optimizer(args, model):
    scale_lr_group = []
    normal_group = []
    scale_lr_group_name, normal_group_name = [], []
    for n, p in model.named_parameters():
        if n.endswith(".weight") and "layernorm" not in n and "embedding" not in n and "lm_head" not in n:
            scale_lr_group.append(p)
            scale_lr_group_name.append(n)
        else:
            normal_group.append(p)
            normal_group_name.append(n)
    bmt.print_rank(scale_lr_group_name, normal_group_name)
    param_groups = [
        {"params": scale_lr_group, "lr": args.lr / model.config.scale_width},
        {"params": normal_group, "lr": args.lr},
    ]

    if args.offload:
        optimizer = bmt.optim.AdamOffloadOptimizer(param_groups, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    else:
        optimizer = bmt.optim.AdamOptimizer(param_groups, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    if not args.force_restart and args.load is not None and args.load_grad:
        exporter.load_optimizer_ckpt(args, optimizer)
        bmt.print_rank("optimizer is loaded!")
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
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

    if args.lr_scheduler == "cosine":
        lr_scheduler = Cosine(
            optimizer,
            start_lr=args.lr,
            warmup_iter=warmup_iters,
            end_iter=end_iter,  # 原来是lr_decay_iter
            num_iter=args.start_step,
            lr_end_restart=args.lr_end_restart,
            resume_no_optimze=args.resume_no_optimze,
        )
    elif args.lr_scheduler == "warmupstabledrop":
        lr_scheduler = WarmupStableDrop(
            optimizer,
            start_lr=args.lr,
            warmup_iter=warmup_iters,
            end_iter=end_iter,  # 原来是lr_decay_iter
            drop_iter=drop_iters,
            num_iter=args.start_step,
            resume_no_optimze=args.resume_no_optimze,
        )
    elif args.lr_scheduler == "warmupstableexp":
        lr_scheduler = WarmupStableExp(
            optimizer,
            start_lr=args.lr,
            warmup_iter=warmup_iters,
            drop_begin=args.drop_begin,  # 原来是lr_decay_iter
            drop_iter=drop_iters,
            drop_rate=args.drop_rate,
            num_iter=args.start_step,
            resume_no_optimze=args.resume_no_optimze,
        )
    else:
        raise NotImplementedError(f"invalid lr_scheduler: {args.lr_scheduler}")
    return lr_scheduler


def setup_model_and_optimizer(args):
    start = time.time()
    tokenizer = get_tokenizer(args)
    bmt.synchronize()
    logger.info("load tokenizer in {:.2f}s".format(time.time() - start))

    start = time.time()
    model = get_model(args)
    logger.info("load model in {:.2f}s".format(time.time() - start))

    start = time.time()
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    logger.info("load lr_scheduler in {:.2f}s".format(time.time() - start))

    return tokenizer, model, optimizer, lr_scheduler


def resume_training(args):
    ckpts = sorted(
        [z for z in chain(*[[os.path.join(x[0], y) for y in x[2]] for x in os.walk(args.save)]) if z.endswith(".pt")],
        reverse=True,
        key=lambda x: (int)(re.search("(\d+).pt", x)[1]),
    )
    # find newest job
    ckpts = sorted(
        ckpts,
        reverse=True,
        key=lambda x: (int)(re.search("job_(\d+)_ckpt", x)[1]),
    )

    if len(ckpts) > 0:
        bmt.print_rank(f"resuming with last checkpoint: {ckpts[0]}")
        args.load = ckpts[0]
        # by default, do not load grad file
        args.load_grad = False
        args.start_step = 0
    else:
        # no ckpts, nothing we can do
        os._exit(1)


def initialize():
    args = get_args(pretrain=True)
    bmt.init_distributed(seed=args.seed, tp_size=args.tp_size)

    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
    if not args.force_restart and args.load is not None:
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
        self.ave_list = self.ave_list[-self.chunk_size :]

    def get(self):
        return sum(self.ave_list) / len(self.ave_list)


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

    model.eval()
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100, reduction="none")

    global_total_task_token = defaultdict(int)  # token by task
    global_total_sum_loss = defaultdict(float)  # loss by task

    last_data, task_names = None, None
    activation_rates = []
    mid_activation_rates = []
    stat_act_window_sizes = [int(s) for s in args.stat_act_window_sizes.split(",")]
    stat_act_window_rates = {f"transfer_rate_{s:02}": [] for s in stat_act_window_sizes}

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
        logits, moe_info = model(
            input=data["inputs"],
            cu_seqlens=data["cu_seqlens"],
            max_seqlen=data["max_seqlen"],
            position_ids=data["position_ids"],
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

        activation_num, total_num = ActivationContext.get_clear_act()
        activation_num = torch.tensor(activation_num).to(device=loss.device, dtype=torch.long)
        total_num = torch.tensor(total_num).to(device=loss.device, dtype=torch.long)
        activation_num = bmt.distributed.all_reduce(activation_num, op="sum").item()
        total_num = bmt.distributed.all_reduce(total_num, op="sum").item()
        activation_rate = round(activation_num * 100 / total_num + 1e-10, 2) if total_num > 0 else 0.
        activation_rates.append(activation_rate)

        mid_activation_num, mid_total_num = ActivationContext.get_clear_moe_intermediate_activation()
        mid_activation_num, mid_total_num = torch.tensor(mid_activation_num, device=loss.device, dtype=torch.long), \
            torch.tensor(mid_total_num, device=loss.device, dtype=torch.long)
        mid_activation_num = bmt.distributed.all_reduce(mid_activation_num, op="sum").item()
        mid_total_num = bmt.distributed.all_reduce(mid_total_num, op="sum").item()
        mid_activation_rate = round(mid_activation_num * 100 / mid_total_num + 1e-10, 2) if mid_total_num > 0 else 0.
        mid_activation_rates.append(mid_activation_rate)

        trans_window_sizes, trans_activation_nums, trans_total_nums = ActivationContext.get_clear_transfer_loss()
        assert len(trans_window_sizes) == len(trans_activation_nums) == len(trans_total_nums)
        transfer_activation_rates = {}
        for ws, act, tot in zip(trans_window_sizes, trans_activation_nums, trans_total_nums):
            act, tot = torch.tensor(act, device=loss.device, dtype=torch.long), torch.tensor(tot, device=loss.device, dtype=torch.long)
            act, tot = bmt.distributed.all_reduce(act, op="sum").item(), bmt.distributed.all_reduce(tot, op="sum").item()
            transfer_activation_rates[f"transfer_rate_{ws:02}"] = round(act * 100 / tot + 1e-10, 2) if tot > 0 else 0.
        transfer_activation_ls = sorted(list(transfer_activation_rates.items()), key=lambda x: x[0])

        load_balance_loss = sum(moe_info["balance_loss"]) / len(moe_info["balance_loss"])
        if not isinstance(load_balance_loss, torch.Tensor):
            load_balance_loss = torch.tensor(load_balance_loss).to(loss)
        global_load_balance_loss = bmt.distributed.all_reduce(load_balance_loss, op="avg").item()

        router_entropy_loss = sum(moe_info["router_entropy"]) / len(moe_info["router_entropy"])
        if not isinstance(router_entropy_loss, torch.Tensor):
            router_entropy_loss = torch.tensor(router_entropy_loss).to(loss)
        global_moe_router_entropy_loss = bmt.distributed.all_reduce(router_entropy_loss, op="avg").item()

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
            "| Iter: {iteration:6d} | loss: {loss:.4f} | "
            "load_balance_loss: {load_balance_loss:.4f} | "
            "router_entropy_loss: {router_entropy_loss:.4f} | "
            "act_rate: {act_rate:.2f} | mid_act_rate: {mid_act_rate:.2f} | "
            + " | ".join([f"{key}: {{{key}:.2f}}" for key, _ in transfer_activation_ls])
        ).format(
            iteration=valid_iteration,
            loss=global_loss,
            load_balance_loss=global_load_balance_loss,
            router_entropy_loss=global_moe_router_entropy_loss,
            act_rate=activation_rate,
            mid_act_rate=mid_activation_rate,
            **transfer_activation_rates,
        ))
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

    ave_activation_rate = sum(activation_rates) / len(activation_rates)
    ave_mid_activation_rate = round(sum(mid_activation_rates) / len(mid_activation_rates), 2)
    ave_transfer_rates = {}
    for key, val in stat_act_window_rates.items():
        ave_transfer_rates[key] = round(sum(val) / len(val), 2)
    save_info_dict = {
        "iteration": iteration,
        "total_sum_loss": total_sum_loss,
        "total_tokens": total_tokens,
        "task_loss": global_total_sum_loss,
        "total_task_token": global_total_task_token,
        "ave_activation_rate": ave_activation_rate,
        "ave_mid_activation_rate": ave_mid_activation_rate,
    }
    save_info_dict.update(ave_transfer_rates)
    assert global_total_sum_loss.keys() == global_total_task_token.keys()
    task_avg = {}
    for key in global_total_sum_loss.keys():
        task_avg[key] = global_total_sum_loss[key] / global_total_task_token[key]
    save_info_dict["task_avg_loss"] = task_avg
    save_every_step_stats(save_info_dict, args.save + "/validation")
    bmt.print_rank(">" * 30, "validation result begin", "<" * 30)
    bmt.print_rank(save_info_dict)
    bmt.print_rank("*" * 10, "activation rate:", ave_activation_rate, "*" * 10)
    bmt.print_rank("*" * 10, "mid activation rate:", ave_mid_activation_rate, "*" * 10)
    for key, val in ave_transfer_rates.items():
        bmt.print_rank("*" * 10, f"{key}:", val, "*" * 10)
    bmt.print_rank(">" * 30, "validation result end", "<" * 30)


def pretrain(
    args,
    tokenizer,
    model: Dragonfly,
    optimizer,
    lr_scheduler: bmt.lr_scheduler.WarmupLRScheduler,
):
    ActivationContext.set_remoe_l1_reg_info(model.config.num_layers, args.l1_reg_coef_init, args.l1_reg_coef_multiplier, args.l1_reg_coef_resume)
    if model.config.moe_routing_strategy in ["relu", "relu_rms"]:
        args.router_aux_loss_coef = 0.
        bmt.print_rank("args.router_aux_loss_coef is set to 0.0 when using ReMoE.")
    if args.valid_only:
        ActivationContext.set_inference_mode(True)

    ave_model_time = ChunkAve(chunk_size=100)
    ave_iter_time = ChunkAve(chunk_size=100)
    accu_iters = args.grad_accum

    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100, reduction="none")
    optim_manager = bmt.optim.OptimManager(
        loss_scale=bmt.world_size(),
        loss_scale_steps=args.loss_scale_steps,
        loss_scale_factor=2,
        max_loss_scale=bmt.world_size(),
        min_loss_scale=bmt.world_size(),
    )
    optim_manager.add_optimizer(optimizer, lr_scheduler)
    start_step = args.start_step
    ActivationContext.set_stat_act_window_sizes(args.stat_act_window_sizes)

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
        )

        if not args.force_restart and args.load is not None and args.load_dataloader_ckpt and args.only_load_model == 0:
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

    try:
        mem_usage, tim_usage = {}, {}
        mem_usage, tim_usage = add_mem_time("before_log", mem_usage, tim_usage)

        for iteration, data in enumerate(DataIterator, start=start_step * accu_iters + 1):
            if args.valid_only or (args.valid_interval > 0 and iteration % args.valid_interval == 0):
                do_evaluation(args, tokenizer, model, iteration)
                if args.valid_only:
                    exit()
            if args.tp_size == 1 or bmt.config["tp_rank"] == 0:
                mixed_indexed_dataset.update_states(data["task_ids"], data["indexes"])

            mem_usage, tim_usage = add_mem_time("init", mem_usage, tim_usage)

            logits, moe_info = model(
                input=data["inputs"],
                cu_seqlens=data["cu_seqlens"],
                max_seqlen=data["max_seqlen"],
                position_ids=data["position_ids"],
                seq_mask=data["seq_mask"],
            )

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
            lm_loss = non_reduced_loss.sum() / _w.sum().float()

            if moe_info["load"] != []:
                moe_load_balance_loss = sum(moe_info["balance_loss"]) / len(moe_info["balance_loss"])
                moe_router_entropy_loss = sum(moe_info["router_entropy"]) / len(moe_info["router_entropy"])
                loss = lm_loss + args.router_aux_loss_coef * moe_load_balance_loss + args.router_ent_loss_coef * moe_router_entropy_loss
                global_lm_loss = bmt.sum_loss(lm_loss).item()
                global_moe_load_balance_loss = bmt.sum_loss(moe_load_balance_loss).item()
                global_moe_router_entropy_loss = bmt.sum_loss(moe_router_entropy_loss).item()
                global_loss = bmt.sum_loss(loss).item()
            else:
                loss = lm_loss
                global_lm_loss = bmt.sum_loss(lm_loss).item()
                global_moe_load_balance_loss = 0
                global_moe_router_entropy_loss = 0
                global_loss = bmt.sum_loss(loss).item()
            loss = loss / accu_iters

            mem_usage, tim_usage = add_mem_time("forward", mem_usage, tim_usage)

            optim_manager.backward(loss)

            activation_num, total_num = ActivationContext.get_clear_act()
            activation_num = torch.tensor(activation_num).to(device=loss.device, dtype=torch.long)
            total_num = torch.tensor(total_num).to(device=loss.device, dtype=torch.long)
            activation_num = bmt.distributed.all_reduce(activation_num, op="sum").item()
            total_num = bmt.distributed.all_reduce(total_num, op="sum").item()
            activation_rate = round(activation_num * 100 / total_num + 1e-10, 2) if total_num > 0 else 0.

            mid_activation_num, mid_total_num = ActivationContext.get_clear_moe_intermediate_activation()
            mid_activation_num, mid_total_num = torch.tensor(mid_activation_num, device=loss.device, dtype=torch.long), \
                torch.tensor(mid_total_num, device=loss.device, dtype=torch.long)
            mid_activation_num = bmt.distributed.all_reduce(mid_activation_num, op="sum").item()
            mid_total_num = bmt.distributed.all_reduce(mid_total_num, op="sum").item()
            mid_activation_rate = round(mid_activation_num * 100 / mid_total_num + 1e-10, 2) if mid_total_num > 0 else 0.

            mem_usage, tim_usage = add_mem_time("backward", mem_usage, tim_usage)

            grad_accum_init_time = tim_usage["init"]
            if iteration % accu_iters == 0 or iteration == (args.train_iters * accu_iters):
                if model.config.moe_routing_strategy in ["relu", "relu_rms"]:
                    target_activation_ratio = round(model.config.moe_top_k * 100 / model.config.moe_num_experts, 2)
                    ActivationContext.step_l1_ref_coef(activation_rate, target_activation_ratio)
                grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, args.clip_grad, norm_type=2)
                optim_manager.step()
                optim_manager.zero_grad()
                mem_usage, tim_usage = add_mem_time("optim", mem_usage, tim_usage)
                model_time = tim_usage["optim"] - grad_accum_init_time
                ave_model_time.record(model_time)
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
            local_total_rate = bmt.sum_loss(local_total_rate).item()
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

            l1_reg_coef = ActivationContext.get_remoe_l1_reg_coef()
            train_info = {
                "time": iter_time,
                "iteration": iteration,
                "real_iteration": iteration // accu_iters,
                "loss": global_loss,
                "activation_rate": activation_rate,
                "lr": lr_scheduler.current_lr,
                "token_max": local_total_rate,
                "token_pass": global_token_pass,
                "throughout": args.max_length * args.batch_size * local_total_rate / ave_iter_time.get() / args.tp_size,
                "grad_norm": grad_norm.item(),
                "mask_max": ((data["targets"] >= 0).sum(-1).float().mean() / args.max_length).item(),
                "task_loss": task_loss_map,
                "total_task_token": global_total_task_token,
                "l1_reg_coef": l1_reg_coef,
            }
            global_token_pass_str = convert_to_k_and_b(global_token_pass)

            time_report_str = "{model_time:.2f}={forward_time:.2f}+{backward_time:.2f}+{optim_time:.2f}".format(model_time=model_time, forward_time=tim_usage['forward']-tim_usage['init'], backward_time=tim_usage['backward']-tim_usage['forward'], optim_time=tim_usage['optim'] - tim_usage['backward'])
            bmt.print_rank(
                (
                    "| Iter: {iteration:6d} | Real_Iter: {real_iter:6d} | loss: {loss:.4f} | lm_loss: {lm_loss:.4f} | moe_balance_loss: {moe_load_balance_loss:.4f} | l1_reg_coef: {l1_reg_coef:.8f} | "
                    "moe_router_entropy_loss: {moe_router_entropy_loss:.4f} | activation_rate: {activation_rate:.2f} | mid_act_rate: {mid_activation_rate:.2f} | lr: {lr:.4e} | model_time: {model_time} | iter_time: {iter_time:.2f} | chunk_ave_time: {chunk_ave_time:.2f} |"
                    + " token/max: {tokenrate:.4f} | mask/max: {maskrate:.4f} | grad_norm: {grad_norm:.4f} | global_token_pass (B):"
                    + "{global_token_pass} | mem_usage {mem_usage} | throughput: {throughput:.2f}"
                ).format(
                    iteration=iteration,
                    real_iter=iteration // accu_iters,
                    loss=global_loss,
                    lm_loss=global_lm_loss,
                    moe_load_balance_loss=global_moe_load_balance_loss,
                    l1_reg_coef=l1_reg_coef,
                    moe_router_entropy_loss=global_moe_router_entropy_loss,
                    activation_rate=activation_rate,
                    mid_activation_rate=mid_activation_rate,
                    lr=lr_scheduler.current_lr,
                    model_time=time_report_str,
                    iter_time=iter_time,
                    chunk_ave_time=ave_iter_time.get(),
                    tokenrate=data["lengths"].float().mean() / args.max_length / args.batch_size,
                    maskrate=(data["targets"] >= 0).sum(-1).float().mean() / args.max_length / args.batch_size,
                    grad_norm=grad_norm.item(),
                    global_token_pass=global_token_pass_str,
                    mem_usage=max([value for key, value in mem_usage.items()]),
                    throughput=args.max_length * args.batch_size * local_total_rate / model_time,
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

            if iteration % (args.inspect_iters * accu_iters) == 0:
                model_inspect = bmt.inspect.inspect_model(model, "*")
                bmt.print_rank(bmt.inspect.format_summary(model_inspect))

            if args.log_dir is not None and bmt.rank() == 0:
                if args.save is not None:
                    save_every_step_stats(train_info, args.save)

            if args.tensorboard is not None and bmt.rank() == 0:
                writer.add_scalar("Loss/train", global_loss, iteration)
                writer.add_scalar("Loss/lm_loss", global_lm_loss, iteration)
                writer.add_scalar("Loss/moe_aux_loss", global_moe_load_balance_loss, iteration)
                writer.add_scalar("Loss/moe_ent_loss", global_moe_router_entropy_loss, iteration)
                writer.add_scalar("Act/train/act_rate", activation_rate, iteration)
                writer.add_scalar("Act/train/l1_reg_coef", l1_reg_coef, iteration)
                writer.add_scalar("Optimizer/lr", lr_scheduler.current_lr, iteration)
                writer.add_scalar("Optimizer/scale", optim_manager.loss_scale, iteration)
                writer.add_scalar("Optimizer/grad_norm", grad_norm.item(), iteration)
                for task_name, loss in task_loss_map.items():
                    if not math.isnan(loss):
                        writer.add_scalar("Loss/train/{}".format(task_name), loss, iteration)

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

    exporter.export(model, mixed_indexed_dataset, tokenizer, optimizer, -1, args, final_save=True)


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
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    bmt.print_rank("finish loading")
    bmt.print_rank(
        "Number of parameter {}, Number of non-e parameter {}".format(
            num_parameters(model), num_non_embedding_parameters(model)
        )
    )
    bmt.print_rank("args: {}".format(args))

    pretrain(args, tokenizer, model, optimizer, lr_scheduler)


if __name__ == "__main__":
    main()
