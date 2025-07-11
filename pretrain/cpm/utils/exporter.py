# COPIED from exporter, for scaling project special use.
# Author: shengdinghu@gmail.com

import glob
import multiprocessing as mp
import re
from copy import deepcopy
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import bmtrain as bmt
import torch
from cpm.training_utils.value_scheduler import AdaptiveLinearScheduler

from cpm.utils.bitset import BitSet

from .log import logger

lock = threading.Lock()


def _save_artifacts(
    model_state, dataloader, tokenizer, opt_state, router_opt_state, value_sche_state, global_step,
    args, model_config, log_ckpt=None, final_save=False
):
    """
    Export model artifacts. Mainly for the purpose of asynchronous saving.
    """
    export_model_dir = args.save if final_save else os.path.join(args.save, f"ckpt_{global_step}")
    os.makedirs(export_model_dir, exist_ok=True)
    base_file_name = f"{args.save_name}-{global_step}" if global_step > -1 else args.save_name
    bmt.print_rank(f"start to export ckpt, save_dir={export_model_dir}, file prefix={base_file_name}")

    # model checkpoint
    ckpt_path = os.path.join(export_model_dir, base_file_name + ".pt")
    # opt 文件仅用于项目内续训，不需要导出为模型版本文件
    opt_path = os.path.join(
        export_model_dir,
        args.save_name + ("-%d.rank-%d.opt" % (global_step, bmt.rank())),
    )
    router_opt_path = os.path.join(
        export_model_dir,
        args.save_name + ("-%d.rank-%d.ropt" % (global_step, bmt.rank())),
    )
    value_sche_path = os.path.join(
        export_model_dir,
        args.save_name + ("-%d.sche" % global_step),
    )

    if bmt.rank() == 0:
        torch.save(model_state, ckpt_path)
        bmt.print_rank(f"Save checkpoint successfully, ckpt file path: {ckpt_path}")
    torch.save(opt_state, opt_path)
    print(f"Save optimizer state successfully, opt file path: {opt_path}")
    torch.save(router_opt_state, router_opt_path)
    print(f"Save optimizer state successfully, router opt file path: {router_opt_path}")
    if bmt.rank() == 0:
        torch.save(value_sche_state, value_sche_path)
        print(f"Save value scheduler state successfully, value scheduler file path: {value_sche_path}")

    del model_state
    del opt_state
    del router_opt_state
    del value_sche_state

    # 保存统计量
    if log_ckpt is not None:
        bmt.print_rank("save log ckpt ...")
        with open(os.path.join(export_model_dir, base_file_name + ".log_ckpt"), "w") as log_ckpt_file:
            json.dump(log_ckpt, log_ckpt_file)

    logger.info(f"Starting saving dataset state. ")

    dataset_ckpt_path = os.path.join(export_model_dir, "dataset_ckpt")
    os.makedirs(dataset_ckpt_path, exist_ok=True)
    if bmt.config["tp_rank"] == 0:
        p_dataset = os.path.join(dataset_ckpt_path, f"dataset_{bmt.rank()}.data")
        dataloader.save_state_dict(p_dataset)

    if bmt.rank() == 0:
        # config 和 vocabs 和模型文件一起存储
        model_config.save_pretrained(export_model_dir)
        try:
            tokenizer.save_pretrained(export_model_dir)
        except:
            bmt.print_rank("No save pretrained method for tokenizer")
            shutil.copy(args.tokenizer_path, export_model_dir)

        # 存储完所有文件后调用
        # if platform_cfg is not None:
        #     platform_cfg.finalize_model_save(export_model_dir, base_file_name)
        # else:
        bmt.print_rank("No platform_cfg, skip finalize_model_save, may be not have .success file")
        logger.info(f"Successfully save model files:  {os.listdir(export_model_dir)}")

    # 每个进程都在export_model_dir写一个.save_done文件，用于判断是否所有进程都保存完毕
    # 直接用常规写文件方式
    os.makedirs(os.path.join(export_model_dir, "save_status"), exist_ok=True)
    with open(os.path.join(export_model_dir, f"save_status/{bmt.rank()}.save_done"), "w") as f:
        f.write("done")

    # 等待所有进程都保存完毕, 不能用synchronize
    if bmt.rank() == 0:
        while True:
            if len(os.listdir(os.path.join(export_model_dir, "save_status"))) == bmt.world_size():
                break
            time.sleep(1)
        bmt.print_rank(f"All saved! Rank 0 Begin to merge dataset ckpt to {dataset_ckpt_path}/dataset_.data")
        merge_dataset_ckpts(export_model_dir, args.parallel_load_datastate // 2)
    else:
        bmt.print_rank(f"rank-{bmt.rank()} done, wait for rank0 to merge dataset ckpt")


def export(
    model: torch.nn.Module,
    dataloader,
    tokenizer,
    optimizer: bmt.optim.AdamOffloadOptimizer,
    router_optimizer: bmt.optim.AdamOffloadOptimizer,
    value_scheduler: AdaptiveLinearScheduler,
    global_step,
    args,
    log_ckpt=None,
    final_save=False,
    async_save=False,
):
    """
    一次 ckpt 保存：
    /{args.save}/
        ├── job_{job_id}_ckpt_{global_step}/  # checkpoint 导出为模型版本时，job_{job_id}_ckpt_{global_step}/ 路径下文件会一起导出，创建一个模型组版本
            ├── config.json
            ├── vocabs.txt
            ├── {args.save_name}-{global_step}.rank-0.opt
            ├── {args.save_name}-{global_step}.rank-n.opt
            ├── {args.save_name}-{global_step}.pt
            ├── {args.save_name}-{global_step}.data
            ├── {args.save_name}-{global_step}.data.json
            ├── {args.save_name}-{global_step}.success
            └── {args.save_name}-{global_step}.log_ckpt

    """

    bmt.synchronize()
    model_state = bmt.store._save_to_rank0(model)
    opt_state = deepcopy(optimizer.state_dict())
    router_opt_state = deepcopy(router_optimizer.state_dict())
    value_sche_state = deepcopy(value_scheduler.state_dict()) if value_scheduler is not None else None
    model_config = model.config

    if async_save:
        # Save artifacts asynchronously
        save_proc = mp.Process(
            target=_save_artifacts,
            args=(
                model_state, dataloader, tokenizer, opt_state, router_opt_state, value_sche_state, global_step,
                args, model_config, log_ckpt, final_save),
        )
        save_proc.start()
    else:
        _save_artifacts(
            model_state, dataloader, tokenizer, opt_state, router_opt_state, value_sche_state, global_step,
            args, model_config, log_ckpt, final_save
        )


def load_model_ckpt(args, model):
    """args.load 是一个到/{args.save}/job_{job_id}_ckpt_{global_step}/ 的路径"""
    if args.load.endswith(".pt"):
        checkpoint_file = args.load
    else:
        # a directory
        load_path = args.load
        checkpoint_files = [file for file in os.listdir(load_path) if file.endswith(".pt")]
        assert len(checkpoint_files) == 1, "None or multiple .pt found in {}".format(load_path)
        checkpoint_file = os.path.join(load_path, checkpoint_files[0])
    bmt.print_rank("args.load is not None, start to load checkpoints from:", checkpoint_file)

    if bmt.rank() == 0 and hasattr(model, "config") and getattr(model.config, "ffn_type", "") == "block_linear":
        ori_ckpt = torch.load(checkpoint_file)
        if ori_ckpt["encoder.layers.0.ffn.ffn.moe_experts.moe_w_in"].ndim == 3:
            for lid in range(model.config.num_layers):
                w_in = ori_ckpt[f"encoder.layers.{lid}.ffn.ffn.moe_experts.moe_w_in"]
                num_expert, dim_expert, dim_model = w_in.shape
                ori_ckpt[f"encoder.layers.{lid}.ffn.ffn.moe_experts.moe_w_in"] = w_in.view(num_expert * dim_expert, dim_model)
                w_out = ori_ckpt[f"encoder.layers.{lid}.ffn.ffn.moe_experts.moe_w_out"]
                ori_ckpt[f"encoder.layers.{lid}.ffn.ffn.moe_experts.moe_w_out"] = w_out.transpose(0, 1).contiguous().view(dim_model, num_expert * dim_expert)
            checkpoint_file = checkpoint_file[:-3] + ".pt.fix"
            torch.save(ori_ckpt, checkpoint_file)
            print("Fix ckpt and saved to:", checkpoint_file, "......")
    if bmt.rank() == 0 and hasattr(model, "config") and getattr(model.config, "ffn_type", "") in ["block_gate1", "block_gate2", "block_relugate2"]:
        ori_ckpt = torch.load(checkpoint_file)
        if "encoder.layers.0.ffn.ffn.weight" in ori_ckpt:
            for lid in range(model.config.num_layers):
                key = f"encoder.layers.{lid}.ffn.ffn.weight"
                del ori_ckpt[key]
            checkpoint_file = checkpoint_file[:-3] + ".pt.fix"
            torch.save(ori_ckpt, checkpoint_file)
            print("Fix ckpt and saved to:", checkpoint_file, "......")

    bmt.load(model, checkpoint_file)
    return model


def load_value_scheduler(args, value_scheduler: AdaptiveLinearScheduler):
    load_path = args.load
    state_files = [file for file in os.listdir(load_path) if file.endswith(".sche")]
    assert len(state_files) == 1, "None or multiple .sche found in {}".format(load_path)
    state_file = os.path.join(load_path, state_files[0])
    bmt.print_rank("args.load is not None, start to load value_scheduler from:", state_file)

    value_scheduler.load_state_dict(torch.load(state_file))
    return value_scheduler


def _legacy_load_optimizer_ckpt(args, optimizer):
    bmt.print_rank("Use legacy optimizer ckpt!")
    if args.load.endswith(".pt"):
        optimizer_path = os.path.dirname(os.path.dirname(args.load))
    else:
        optimizer_path = os.path.dirname(args.load)
    bmt.print_rank(os.listdir(optimizer_path))
    start = time.time()
    bmt.print_rank(
        "{}".format(
            sum(
                [
                    1
                    if i.find(".opt") != -1 and i.find("-{}.rank".format(args.start_step % (args.save_iters * 5))) != -1
                    else 0
                    for i in os.listdir(optimizer_path)
                ]
            )
        )
    )
    if (
        sum(
            [
                1
                if i.find(".opt") != -1 and i.find("-{}.rank".format(args.start_step % (args.save_iters * 5))) != -1
                else 0
                for i in os.listdir(optimizer_path)
            ]
        )
        == bmt.world_size()
    ):
        pattern = "-{}.rank-{}.opt".format(args.start_step % (args.save_iters * 5), bmt.rank())
        bmt.print_rank("Will load opt that matches pattern: {}".format(pattern))
        for file_name in os.listdir(optimizer_path):
            if file_name.find(pattern) != -1:
                bmt.print_rank("start to load grad ckpt {}".format(file_name))
                states = torch.load(os.path.join(optimizer_path, file_name))
                optimizer.load_state_dict(states)
    logger.info("load grad in {:.2f}s".format(time.time() - start))
    return optimizer


def load_optimizer_ckpt(args, optimizer, is_router: bool = False):
    suffix = ".ropt" if is_router else ".opt"
    if args.load.endswith(".pt"):
        optimizer_path = os.path.dirname(args.load)
    else:
        # a directory
        optimizer_path = args.load
    start = time.time()
    opt_num = sum(
        [1 if re.search(r"-{}.rank-\d+{}".format(args.start_step, suffix), i) else 0 for i in
         os.listdir(optimizer_path)]
    )

    bmt.print_rank(f"Opt file num: {opt_num}")
    if opt_num == 0:
        return _legacy_load_optimizer_ckpt(args, optimizer)

    # if opt_num == bmt.world_size():
    r = bmt.rank() % (args.grad_ckpt_num)
    assert opt_num == args.grad_ckpt_num
    file_name = os.path.join(
        optimizer_path,
        args.save_name + "-{}.rank-{}{}".format(args.start_step, r, suffix),
    )

    if os.path.exists(file_name):
        print("rank {} start to load grad ckpt {}".format(bmt.rank(), file_name))
        states = torch.load(file_name)
        optimizer.load_state_dict(states)
    logger.info("load grad in {:.2f}s".format(time.time() - start))
    return optimizer


def load_dataloader_ckpt(args, mixed_dataset):
    load_success = _load_distributed_dataset_state(args, mixed_dataset)
    if not load_success:
        logger.info("load from distributed data state dict fail, try to load from single data state_dict")
        _load_dataloader_ckpt(args, mixed_dataset)


def _load_dataloader_ckpt(args, mixed_dataset):
    """args.load 是一个到/{args.save}/job_{job_id}_ckpt_{global_step}/ 的路径"""
    if args.load.endswith(".pt"):
        load_path = os.path.dirname(args.load)
    else:
        load_path = args.load
    dataset_states_path = [file for file in os.listdir(load_path) if file.endswith(".data")]
    assert len(dataset_states_path) == 1, "None or multiple .data found in {}, file list: {}".format(
        load_path, dataset_states_path
    )
    dataset_states_path = dataset_states_path[0]
    dataset_states_path = os.path.join(load_path, dataset_states_path)
    bmt.print_rank("args.load is not None, start to load data ckpt from " + dataset_states_path)
    dataset_states = torch.load(dataset_states_path)
    missing = mixed_dataset.load_state_dict(dataset_states)
    if len(missing) > 0:
        bmt.print_rank("missing keys when loading dataset states: ", missing)


def load_trace_ckpt(args, dataloader):
    """args.load 是一个到/{args.save}/job_{job_id}_ckpt_{global_step}/ 的路径"""
    return dataloader


def load_log_ckpt(args):
    if args.load.endswith(".pt"):
        load_path = os.path.dirname(args.load)
    else:
        load_path = args.load
    log_ckpt_paths = [file for file in os.listdir(load_path) if file.endswith(".log_ckpt")]
    assert len(log_ckpt_paths) <= 1, "Multiple .data found in {}".format(load_path)
    if len(log_ckpt_paths) == 0:
        bmt.print_rank("No log ckpt is found in {}".format(load_path))
        return {}
    log_ckpt_path = os.path.join(load_path, log_ckpt_paths[0])
    bmt.print_rank("args.load is not None, start to load log ckpt from " + log_ckpt_path)
    with open(log_ckpt_path, "r") as log_ckpt_file:
        log_ckpt = json.load(log_ckpt_file)
    return log_ckpt


def _load_distributed_dataset_state(args, mixed_dataset):
    rank = bmt.rank()
    logger.info(f"rank-{rank} -> [start]loading dataset states")
    if args.load.endswith(".pt"):
        load_dir = os.path.dirname(args.load)
    else:
        load_dir = args.load

    p_datasets = sorted(glob.glob(os.path.join(load_dir, "dataset_ckpt/dataset_*.data")))
    if len(p_datasets) == 0:  # 向后兼容
        bmt.print_rank("load_from_orginal_dataset_ckpt_folder")
        p_datasets = sorted(glob.glob(os.path.join(load_dir, "dataset_*.data")))

    all_state_dict = dict()

    def load_and_aggregate(p_dataset):
        """Map func for loading and aggregating dataset states to all_state_dict"""

        def new_key_init(all_state_dict, key, state_dict):
            all_state_dict[key] = {}
            for second_key in state_dict[key]:
                if second_key == "used":
                    all_state_dict[key]["used"] = BitSet(1024)
                    all_state_dict[key]["used"].update(state_dict[key]["used"])
                else:
                    all_state_dict[key][second_key] = state_dict[key][second_key]

        def load_state_dict_with_npy(p_dataset):
            state_dict = torch.load(p_dataset)
            for key in state_dict:
                if isinstance(state_dict[key]["used"], str):
                    if os.path.basename(state_dict[key]["used"]) == state_dict[key]["used"]: # 如果只有文件名
                        state_dict[key]["used"] = BitSet.load(os.path.join(load_dir, "dataset_ckpt", os.path.basename(state_dict[key]["used"])))
                    else: # 如果是完整路径（向后兼容）
                        state_dict[key]["used"] = BitSet.load(state_dict[key]["used"])
            return state_dict

        print(f"Loading {p_dataset}...")
        state_dict = load_state_dict_with_npy(p_dataset)
        dataset_locks = {}
        for key in state_dict.keys():
            if key not in dataset_locks:
                with lock:
                    if key not in dataset_locks:
                        dataset_locks[key] = threading.Lock()

            with dataset_locks[key]:
                if key in all_state_dict:
                    if all_state_dict[key]["exhausted"]:
                        continue
                    elif state_dict[key]["exhausted"]:
                        all_state_dict[key]["exhausted"] = True
                        all_state_dict[key]["used"] = BitSet(1024)
                    else:
                        all_state_dict[key]["used"].update(state_dict[key]["used"])
                else:
                    new_key_init(all_state_dict, key, state_dict)
        bmt.print_rank(f"[done]loaded dataset states: {p_dataset}")

    if p_datasets:
        rank = bmt.rank()
        lst_time = time.time()
        if rank == 0:
            with ThreadPoolExecutor(max_workers=args.parallel_load_datastate) as executor:
                executor.map(load_and_aggregate, p_datasets)

        logger.info(
            f"rank-{rank} -> load dataset from {len(p_datasets)} .data files. Time: {time.time() - lst_time:.2f}s"
        )
        # Broadcast the tensor to other process
        lst_time = time.time()
        all_state_dict = bmt.store.broadcast_object(all_state_dict, comm=bmt.config["comm"], src=0)
        logger.info(f"rank-{rank} -> broadcast dataset from rank-0 to other ranks. Time: {time.time() - lst_time:.2f}s")
        lst_time = time.time()

        missing_keys = mixed_dataset.load_state_dict(all_state_dict)
        logger.info(f"rank-{rank} -> load mixed dataset state dict. Time: {time.time() - lst_time:.2f}s")
        if missing_keys:
            logger.info(
                f"rank-{rank} -> load dataset from {len(p_datasets)} .data files with {os.path.join(load_dir, 'dataset_ckpt', 'dataset*.pt')} : {p_datasets}, missing tasks: {missing_keys}"
            )
        else:
            state_info = {
                k: {
                    "ave_tokens": s.get("ave_tokens", -1),
                    "set_info": "mem:{}|density:{:.4f}|len:{}".format(
                        s["used"].memory_usage(), s["used"].density(), len(s["used"])
                    ),
                }
                for k, s in all_state_dict.items()
            }
            logger.info(
                f"rank-{rank} -> load dataset from {len(p_datasets)} files with {os.path.join(load_dir, 'dataset*.pt')}. Info: {state_info}"
            )
        return True
    else:
        logger.info(f"No dataset*.data found. p_datasets: {p_datasets}")
        return False


import json
import os
from collections import OrderedDict


def flatten_stats(stats, parent_key="", separator="/"):
    items = []
    for key, value in stats.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_stats(value, new_key, separator).items())
        if isinstance(value, list):
            items.append((new_key, json.dumps(value)))
        else:
            items.append((new_key, value))
    return OrderedDict(items)


def save_every_step_stats(stats, path, flatten=True, max_file_size=19):
    if flatten:
        stats = flatten_stats(stats)

    os.makedirs(os.path.join(path, "train_stats/"), exist_ok=True)

    # Function to get the current file ID and size
    def get_current_file_id_and_size(path):
        id = 0
        while True:
            file_path = os.path.join(path, f"train_stats/{id}.jsonl")
            if not os.path.exists(file_path):
                return id, 0
            else:
                size = os.path.getsize(file_path)
                if size > max_file_size * 1024 * 1024:  # Size in bytes (20 MB)
                    id += 1
                else:
                    return id, size

    # Get the current file id and its size
    current_id, file_size = get_current_file_id_and_size(path)

    # Generate the file path
    file_path = os.path.join(path, f"train_stats/{current_id}.jsonl")

    # Write the flattened stats to the file
    with open(file_path, "a") as json_file:
        json_file.write(json.dumps(stats) + "\n")


def merge_dataset_ckpts(load_dir, parallel_load_datastate):
    p_datasets = sorted(glob.glob(os.path.join(load_dir, "dataset_ckpt/dataset_*.data")))
    p_datasets = [x for x in p_datasets if "dataset_ckpt/dataset_.data" not in x]

    bmt.print_rank(f"Files before merge (total num {len(p_datasets)}): {p_datasets}")

    all_state_dict = dict()

    def load_and_aggregate(p_dataset):
        """Map func for loading and aggregating dataset states to all_state_dict"""

        def new_key_init(all_state_dict, key, state_dict):
            all_state_dict[key] = {}
            for second_key in state_dict[key]:
                if second_key == "used":
                    all_state_dict[key]["used"] = BitSet(1024)
                    all_state_dict[key]["used"].update(state_dict[key]["used"])
                else:
                    all_state_dict[key][second_key] = state_dict[key][second_key]

        def load_state_dict_with_npy(p_dataset):
            state_dict = torch.load(p_dataset)
            for key in state_dict:
                if isinstance(state_dict[key]["used"], str):
                    state_dict[key]["used"] = BitSet.load(os.path.join(load_dir, "dataset_ckpt", os.path.basename(state_dict[key]["used"])))
            return state_dict

        print(f"Loading {p_dataset}...")
        state_dict = load_state_dict_with_npy(p_dataset)
        dataset_locks = {}

        for key in state_dict.keys():
            if key not in dataset_locks:
                with lock:
                    if key not in dataset_locks:
                        dataset_locks[key] = threading.Lock()
            with dataset_locks[key]:
                if key in all_state_dict:
                    if all_state_dict[key]["exhausted"]:
                        continue
                    elif state_dict[key]["exhausted"]:
                        all_state_dict[key]["exhausted"] = True
                        all_state_dict[key]["used"] = BitSet(1024)
                    else:
                        all_state_dict[key]["used"].update(state_dict[key]["used"])
                else:
                    new_key_init(all_state_dict, key, state_dict)
        del state_dict
        print(f"Merged {p_dataset}...")

    if p_datasets:
        # with ThreadPoolExecutor(max_workers=args.parallel_load_datastate) as executor:
        with ThreadPoolExecutor(max_workers=parallel_load_datastate) as executor: # smaller than normal load to avoid OOM

            executor.map(load_and_aggregate, p_datasets)
        # load_and_aggregate(p_datasets[0])

        # Broadcast the tensor to other process
        save_path = os.path.join(load_dir, "dataset_ckpt", "dataset_.data")

        # save_path = os.path.join("dataset_.data")
        for key in all_state_dict:
            npy_path = all_state_dict[key]["used"].save(save_path)
            all_state_dict[key]["used"] = npy_path
        bmt.print_rank(f"All state_dict after merge {all_state_dict}")
        torch.save(all_state_dict, save_path)

        # Find all files that match the pattern
        files_to_remove = glob.glob(os.path.join(load_dir, "dataset_ckpt", "dataset_*.data*"))

        # Remove the files
        for file in files_to_remove:
            if "dataset_.data" not in file:
                os.remove(file)

        files_after_merge = os.listdir(os.path.join(load_dir, "dataset_ckpt"))
        bmt.print_rank(f"Files after merge: {files_after_merge}")
