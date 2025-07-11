#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2023 AI, ZHIHU Inc. (zhihu.com)
#
# @author: ouzebin
# @date: 2023/09/27


import copy
import ctypes
import functools
import importlib
import importlib.util
import json
import logging
import multiprocessing as mp
import os
import random
from collections import defaultdict
from collections import OrderedDict
from multiprocessing import Lock
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import bmtrain as bmt
import numpy as np
import requests
import torch
from numpy.typing import NDArray
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_fixed
from tenacity import wait_random

from cpm.dataset import PrefetchDecodeDataset
from cpm.utils.bitset import BitSet
from cpm.utils.vdc_sampling import van_der_corput
from cpm.utils.vdc_sampling import van_der_corput_sampling_gen

from .flask_ps import app as flask_ps

logger = logging.getLogger(__name__)
IGNORE_TGT = -100


def load_dataset_cfgs(cfg_path, cfg_json_str=None):
    if cfg_json_str is not None:
        cfgs = json.loads(cfg_json_str)
    else:
        with open(cfg_path, "r", encoding="utf-8") as fin:
            cfgs = json.load(fin)
    transform_basedir = os.path.dirname(os.path.abspath(cfg_path))

    path_dict = None
    platform_config_path = os.getenv("PLATFORM_CONFIG_PATH")
    try:
        with open(platform_config_path, "r") as f:
            platform_cfg = json.load(f)
        path_dict = platform_cfg["dataset_map"]
        if bmt.rank() == 0:
            logger.info(f"Loaded jeeves platform config from '{platform_config_path}', update dataset paths...")
    except Exception as e:
        if bmt.rank() == 0:
            logger.info(f"Failing to load jeeves platform config '{platform_config_path}', error message:\n{str(e)}")

    task_name2dataset_name = dict()
    for idx, cfg in enumerate(cfgs):
        assert "dataset_name" in cfg and isinstance(cfg["dataset_name"], str)
        assert "task_name" in cfg and isinstance(cfg["task_name"], str)
        # to be delibrately annoying :)
        if cfg["task_name"] in task_name2dataset_name:
            raise ValueError(
                f"task_name '{cfg['task_name']}' in dataset '{cfg['dataset_name']}'"
                f"has already been used in '{task_name2dataset_name[cfg['task_name']]}'."
            )
        task_name2dataset_name[cfg["task_name"]] = cfg["dataset_name"]

        assert "path" in cfg and isinstance(cfg["path"], str)
        # if path_dict is not None:
        # cfg["path"] = os.path.join(path_dict[cfg["dataset_name"]], cfg["path"])

        # dealing with optional configs
        if "weight" in cfg:
            assert isinstance(cfg["weight"], (float, int))
        else:
            cfg["weight"] = 1.0

        if "oversize_rule" in cfg:
            assert cfg["oversize_rule"] in ("drop", "head", "segment")
        else:
            cfg["oversize_rule"] = "segment"

        if "transforms" in cfg:
            assert isinstance(cfg["transforms"], str)
            # dealing with relative path
            if not cfg["transforms"].startswith("/"):
                cfg["transforms"] = os.path.join(transform_basedir, cfg["transforms"])
            if not cfg["transforms"]:
                cfg["transforms"] = None
        else:
            cfg["transforms"] = None

        if "incontext_weight" in cfg:
            assert isinstance(cfg["incontext_weight"], (list, tuple))
        else:
            cfg["incontext_weight"] = [1.0]
        cfg["id"] = idx
        # dataset and iterator will be built
    return cfgs


def data2ids(data, tokenizer, max_length):
    text = "\n".join(
        [
            data.get("title", "").strip(),
            data.get("question", "").strip(),
            data.get("answer", "").strip(),
            data.get("abstract", "").strip(),
            data.get("text", "").strip(),
            data.get("code", "").strip(),
        ]
    ).strip()

    if not text:
        logger.warning(f"Warning: skip invalid sample without valid fields: {data}")
        yield from ()
        return
    # suppress the annoying warning from tokenizer
    ids = (
        [tokenizer.bos_token_id]
        + tokenizer.encode(text, max_length=int(1e12), truncation=True)
        + [tokenizer.eos_token_id]
    )
    src_ids = ids[0:-1]
    tgt_ids = ids[0:-1]  # do not shift because it'll be shifted during loss calculation.

    if len(src_ids) > max_length:
        for st in range(0, len(src_ids), max_length):
            yield src_ids[st : st + max_length], tgt_ids[st : st + max_length]
    else:
        yield src_ids, tgt_ids


def cricket_data2ids(data, tokenizer, max_length: int, oversize_rule="segment", do_compact=False):
    assert oversize_rule in ("drop", "head", "segment")
    if data is None:
        yield from ()
        return
    if "output" not in data or not data["output"]:
        yield from ()
        return
    if "input" not in data:
        data["input"] = ""

    src_ids = [tokenizer.bos_token_id]
    tgt_ids = []
    has_input = False
    is_segment_reenter = False

    # Use incremental tokenization to avoid waiting for a long document
    MAX_CHUNK_LENGTH = max_length * 10
    for part in ("input", "output"):
        l, r = 0, min(MAX_CHUNK_LENGTH, len(data[part]))
        while l < len(data[part]):
            current_slice = data[part][l:r]
            if not current_slice:
                break
            token_ids = tokenizer.encode(current_slice, add_special_tokens=False)
            if part == "input":
                if len(token_ids) > 0:
                    has_input = True
                if len(token_ids) >= max_length - 2:  # input len must < max_length
                    yield from ()
                    return
                src_ids.extend(token_ids)
                tgt_ids.extend([IGNORE_TGT] * len(token_ids))
                l = r
                r = min(len(data[part]), l + MAX_CHUNK_LENGTH)
            else:
                if len(token_ids) + len(tgt_ids) >= max_length:
                    if oversize_rule == "drop":
                        yield from ()
                        return
                    elif oversize_rule == "head":
                        selected_token_ids = token_ids[: max_length - len(src_ids) + 1]
                        src_ids.extend(selected_token_ids[:-1])
                        tgt_ids.extend(selected_token_ids)
                        assert len(src_ids) == len(tgt_ids), f"len (src, tgt): ({len(src_ids)}, {len(tgt_ids)})"
                        yield src_ids[:max_length], tgt_ids[:max_length]
                        return
                    elif oversize_rule == "segment":
                        instruction_rest_space = max_length - 1 - len(token_ids)
                        if has_input:  # is instruction data
                            if (
                                do_compact
                                and len(src_ids) >= 128  # avoid too short instruction info lost
                                and instruction_rest_space / len(src_ids) > 0.8
                            ):  # can be squeezed into max length
                                inputs_len = len(src_ids)
                                keep_len = instruction_rest_space // 2
                                src_ids = src_ids[:keep_len] + src_ids[inputs_len - keep_len :]
                                tgt_ids = [IGNORE_TGT] * (len(src_ids) - 1)
                                src_ids.extend(token_ids)
                                tgt_ids.extend(token_ids)
                                tgt_ids.append(tokenizer.eos_token_id)
                                assert len(src_ids) < max_length, f"len src_ids: {len(src_ids)}"
                                assert len(src_ids) == len(tgt_ids), f"len (src, tgt): ({len(src_ids)}, {len(tgt_ids)})"
                                yield src_ids, tgt_ids
                            else:  # else use head rule
                                selected_token_ids = token_ids[: max_length - len(src_ids) + 1]
                                src_ids.extend(selected_token_ids[:-1])
                                tgt_ids.extend(selected_token_ids)
                                assert len(src_ids) == len(tgt_ids), f"len (src, tgt): ({len(src_ids)}, {len(tgt_ids)})"
                                yield src_ids[:max_length], tgt_ids[:max_length]
                            return
                        else:  # normal segment
                            selected_token_ids = token_ids[: max_length - len(src_ids) + 1]
                            src_ids.extend(selected_token_ids)
                            tgt_ids.extend(selected_token_ids)
                            assert len(src_ids) == max_length + 1, f"len src_ids: {len(src_ids)}"
                            assert len(tgt_ids) == max_length, f"len tgt_ids: {len(tgt_ids)}"
                            yield src_ids[:max_length], tgt_ids[:max_length]
                            src_ids = src_ids[max_length:]
                            tgt_ids = tgt_ids[max_length:]
                            # sliding input str window
                            consumed_str = tokenizer.decode(selected_token_ids)
                            l += len(consumed_str)
                            r = min(len(data[part]), l + MAX_CHUNK_LENGTH)
                            is_segment_reenter = True
                else:
                    if (is_segment_reenter and len(token_ids) > 8) or (
                        not is_segment_reenter and len(token_ids) > 0
                    ):  # is segmented LM data
                        src_ids.extend(token_ids)
                        tgt_ids.extend(token_ids)
                        tgt_ids.append(tokenizer.eos_token_id)
                        assert len(src_ids) == len(tgt_ids), f"len (src, tgt): ({len(src_ids)}, {len(tgt_ids)})"
                        yield src_ids, tgt_ids
                    else:
                        yield from ()
                    return


class SegmentedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        cfg,
        tokenizer,
        max_length=1024,
        transform_func=None,
        nthreads=1,
        prefetch_slice=3,
        slice_size=500,
        do_compact=False,
        is_local=False,
    ):
        def get_full_qualified_name(func):
            module_name = func.__module__
            qual_name = func.__qualname__
            return f"{module_name}.{qual_name}"

        super(SegmentedDataset, self).__init__()
        self.segment = functools.partial(
            cricket_data2ids, tokenizer=tokenizer, max_length=max_length, do_compact=do_compact
        )
        self.cfg = cfg
        self.max_length = max_length
        self.nthreads = nthreads
        self.transform_func = transform_func
        self.prefetch_slice = prefetch_slice
        self.slice_size = slice_size
        self.abs_weight = cfg.get("abs_weight", None)
        self.task_name = cfg["task_name"]
        self.dataset_name = cfg["dataset_name"]
        self.oversize_rule = cfg["oversize_rule"]
        self.dataset = PrefetchDecodeDataset(path=cfg["path"], allow_repeat=cfg.get("allow_repeat", False))
        self.exhausted = False
        self.iterator = None

        self.counter = 0
        self.allow_repeat = cfg.get("allow_repeat", True)
        self.used = set()
        self.is_local = is_local
        self.port_offset = random.randint(1, 8000) + 1000
        if is_local or bmt.rank() == 0:
            addr = os.environ["MASTER_ADDR"]
            port = int(os.environ["MASTER_PORT"]) + self.port_offset
            _avg_tokens = cfg.get("ave_tokens_per_line", -1)
            _avg_tokens = cfg.get("avg_tokens", _avg_tokens)
            requests.post(f"http://{addr}:{port}/avg_tokens/{self.task_name}?action=set&length={_avg_tokens}")

    @retry(stop=stop_after_attempt(3), wait=wait_random(5, 20))
    def set_avg_tokens(self, avg_tokens):
        addr = os.environ["MASTER_ADDR"]
        port = int(os.environ["MASTER_PORT"]) + self.port_offset
        url = f"http://{addr}:{port}/avg_tokens/{self.task_name}?action=set&length={avg_tokens}"
        response = requests.post(url)
        if response.status_code != 200:
            self.reset_port_offset()
            print(f"Failed to set avg_tokens for task {self.task_name}, request url: {url}")
            raise RuntimeError(f"Failed to set avg_tokens for task {self.task_name}, request url: {url}")

    @retry(stop=stop_after_attempt(3), wait=wait_random(5, 20))
    def update_avg_tokens_by_ema(self, length):
        addr = os.environ["MASTER_ADDR"]
        port = int(os.environ["MASTER_PORT"]) + self.port_offset
        url = f"http://{addr}:{port}/avg_tokens/{self.task_name}?action=update&length={length}"
        response = requests.post(url)
        if response.status_code != 200:
            self.reset_port_offset()
            print(f"Failed to update avg_tokens for task {self.task_name}, request url: {url}")
            raise RuntimeError(f"Failed to update avg_tokens for task {self.task_name}, request url: {url}")

    @property
    @retry(stop=stop_after_attempt(3), wait=wait_random(5, 20))
    def avg_tokens(self):
        addr = os.environ["MASTER_ADDR"]
        port = int(os.environ["MASTER_PORT"]) + self.port_offset
        url = f"http://{addr}:{port}/avg_tokens/{self.task_name}"
        response = requests.get(url)
        if response.status_code != 200:
            self.reset_port_offset()
            print(f"Failed to get avg_tokens for task {self.task_name}, request url: {url}")
            raise RuntimeError(f"Failed to get avg_tokens for task {self.task_name}, request url: {url}")
        data = response.json()
        return data["avg_tokens"]

    def reset_port_offset(self):
        self.port_offset = random.randint(1, 8000) + 1000

    def size(self):
        return self.dataset.size()

    def __iter__(self):
        self.iterate()
        return self

    def reset(self):
        self.exhausted = False
        if self.iterator is not None:
            self.iterator.close()
            self.iterator = None
        self.used = BitSet()
        print("Rank {}, Reset dataset:{} done.".format(bmt.rank(), self.dataset_name))

    def transform(self, data: dict) -> dict:
        weight = np.array(self.cfg["incontext_weight"], dtype=np.float32)
        weight = weight / weight.sum()
        num_incontext = np.random.choice(weight.shape[0], p=weight)
        return self.transform_func(data, num_incontext, random.Random())

    def segment_iterate(self, sample_iter):
        for index, data in self.dataset.sliced_iterate(self.nthreads, self.prefetch_slice, self.slice_size, self.used):
            for src_ids, tgt_ids in self.segment(self.transform(data)):
                self.update_avg_tokens_by_ema(len(src_ids))  # 0 for input ids
                yield src_ids, tgt_ids, index

    def iterate(self):
        # make the dataset itself an iterator
        sample_iter = self.dataset.sliced_iterate(self.nthreads, self.prefetch_slice, self.slice_size, self.used)
        self.iterator = self.segment_iterate(sample_iter)

    def __next__(self):
        # advance the task iterator
        if self.iterator is None:
            self.iterate()
        try:
            return next(self.iterator)
        except StopIteration:
            self.exhausted = True
            return None

    def load_state_dict(self, state_dict):
        if state_dict.get("exhausted", False):
            self.exhausted = True
            self.used = BitSet()
        else:
            used = state_dict.get("used", BitSet())
            if len(used) == len(self.dataset):
                self.exhausted = True
                self.used = BitSet()
            else:
                self.exhausted = False
                self.used = used
        _avg_tokens = state_dict.get("ave_tokens", -1)
        _avg_tokens = state_dict.get("avg_tokens", _avg_tokens)
        if self.avg_tokens == -1 or self.avg_tokens < _avg_tokens:
            self.set_avg_tokens(_avg_tokens)

    def state_dict(self):
        if len(self.used) == len(self.dataset):
            return dict(exhausted=True, used=BitSet(), avg_tokens=self.avg_tokens)
        else:
            return dict(exhausted=False, used=self.used, avg_tokens=self.avg_tokens)

    def update_state(self, indice):
        self.used.update(indice)


class MixedIndexedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        cfg_path: str,
        cfg_json_str,
        tokenizer,
        max_length,
        weight_by_size: bool = True,
        nthreads=5,
        prefetch_slice=100,
        parallel_loading=False,
        vdc_sampling=False,
        update_weights_frequency=1,
        seed=42,
    ):
        if bmt.rank() == 0:
            port = int(os.environ["MASTER_PORT"]) + 2188
            self.flask_ps_proc = mp.Process(target=flask_ps.run, kwargs={"host": "0.0.0.0", "port": port})
            self.flask_ps_proc.start()
        super(MixedIndexedDataset, self).__init__()
        self.set_seed(seed + bmt.rank())
        self.weight_by_size = weight_by_size
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.path2transform = dict()
        self.task_dict = OrderedDict()
        self.nthreads = nthreads
        self.prefetch_slice = prefetch_slice
        # useful for indexing
        self.tasks = []
        self.names = []
        # ending of iteration
        self.remain = 0
        self.max_length = max_length
        self.vdc_sampling = vdc_sampling
        if self.vdc_sampling:
            self._vdc_values = [van_der_corput(i) for i in range(100000)]
            self.vdc_gen = van_der_corput_sampling_gen(self._vdc_values)

        self.update_weights_frequency = update_weights_frequency

        self.path2transform = dict()

        cfgs = load_dataset_cfgs(cfg_path, cfg_json_str)
        _sum_weight = sum([cfg["abs_weight"] for cfg in cfgs])
        _weights = {cfg["task_name"]: cfg["abs_weight"] / _sum_weight for cfg in cfgs}
        bmt.print_rank("Absolute Weight of DataSet {}".format(_weights))

        if parallel_loading:
            self.parallel_load(cfgs, max_workers=None)
        else:
            self.sequential_load(cfgs)

        self.weights = None
        self.update_weights()

    def set_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def load_task(self, cfg):
        logger.info(f"Loading {cfg['path']}")
        transform_func = self.get_transform_func(cfg["task_name"], cfg["transforms"])
        task = SegmentedDataset(
            cfg,
            self.tokenizer,
            self.max_length,
            transform_func=transform_func,
            nthreads=self.nthreads,
            prefetch_slice=self.prefetch_slice,
            do_compact=cfg.get("do_compact", False),  # dataset level do_compact
        )
        return task

    def sequential_load(self, cfgs):
        self.cfgs = cfgs
        for cfg in cfgs:
            # python3.7 and later preserves insertion order to dictionary
            logger.info(f"Loading {cfg['path']}")

            transform_func = self.get_transform_func(cfg["task_name"], cfg["transforms"])
            task = SegmentedDataset(
                cfg,
                self.tokenizer,
                self.max_length,
                transform_func=transform_func,
                nthreads=self.nthreads,
                prefetch_slice=self.prefetch_slice,
                do_compact=cfg.get("do_compact", False),  # dataset level do_compact
            )
            self.task_dict[task.task_name] = task
            self.tasks.append(task)
            self.names.append(task.task_name)
            self.remain += 1
        self.weights = None
        self.update_weights()

    def load_state_dict(self, state_dict):
        missing_keys = []
        for name, task in self.task_dict.items():
            if name in state_dict:
                task.load_state_dict(state_dict[name])
            else:
                missing_keys.append(name)
        self.update_weights()
        return missing_keys

    def save_state_dict(self, path):
        state_dict = {}
        for name, task in self.task_dict.items():
            _state_dict = task.state_dict()
            if isinstance(_state_dict["used"], BitSet):
                bitset = _state_dict["used"]
                _file_name = bitset.save(path)
                _state_dict["used"] = _file_name
                state_dict[name] = _state_dict
            else:
                state_dict[name] = task.state_dict()
        torch.save(state_dict, path)
        logger.info("Dataset state saved")

    def update_states(self, task_ids, indice):
        is_dict = isinstance(indice, dict)
        uniq = torch.unique(task_ids)
        for idx in uniq:
            idx = idx.item()
            indexes = indice[idx] if is_dict else indice[task_ids == idx].tolist()
            self.tasks[idx].update_state(indexes)

    def get_transform_func(self, module_name: str, transform_script_path):
        if transform_script_path is None:
            # allow null transform
            return lambda data, num_incontext, rand: data
        if "/" in module_name:
            module_name = "cpm_live.transforms.{}".format(module_name.split("/")[-1])
        else:
            module_name = "cpm_live.transforms.{}".format(module_name)
        if transform_script_path not in self.path2transform:
            # loader = importlib.machinery.SourceFileLoader(module_name, transform_script_path)
            # spec = importlib.util.spec_from_loader(loader.name, loader)
            spec = importlib.util.spec_from_file_location(module_name, transform_script_path)
            if spec is None:
                raise RuntimeError("Spec is none! {}".format(module_name))
            mod = importlib.util.module_from_spec(spec)
            self.path2transform[transform_script_path] = {
                "module": mod,
                "last_mtime": 0,
            }
        transform_script_info = self.path2transform[transform_script_path]
        curr_mtime = float(os.path.getmtime(transform_script_path))
        if curr_mtime > transform_script_info["last_mtime"]:
            transform_script_info["last_mtime"] = curr_mtime
            # load module
            spec.loader.exec_module(transform_script_info["module"])
        transform_func = getattr(transform_script_info["module"], "transform", None)
        if transform_func is None:
            raise NotImplementedError("Find no transform funcion in script '{}'".format(transform_script_path))
        return transform_func

    def update_weights(self):
        task0 = self.tasks[0]
        if task0.abs_weight is not None:  # 这一份config是指定绝对比例的
            weights = []
            for task in self.tasks:
                if task.exhausted:
                    weights.append(0)
                else:
                    if task.avg_tokens == -1:
                        weights.append(task.abs_weight / self.max_length)
                    else:
                        weights.append(task.abs_weight / task.avg_tokens)
            weights = np.array(weights)
        else:
            weights = np.array([0 if task.exhausted else task.weight for task in self.tasks])
            if self.weight_by_size:
                sizes = np.array([task.size() for task in self.tasks], dtype=np.float32)
                weights *= sizes
        self.weights = weights / weights.sum()

    def __iter__(self):
        for task in self.tasks:
            task.iterate()
        return self

    def __next__(self):
        step = 1
        while True:
            if self.remain == 0:
                print("Rank {}, All task exhaust !!!!".format(bmt.rank()))
                raise StopIteration
            if self.vdc_sampling:
                idx = next(self.vdc_gen)(self.weights)
            else:
                idx = np.random.choice(len(self.weights), p=self.weights)

            data = next(self.tasks[idx])
            if step % self.update_weights_frequency == 0:
                self.update_weights()
            if data is None:
                if self.tasks[idx].allow_repeat:
                    # _runtime_ave = self.tasks[idx].avg_tokens
                    print("Rank {}, dataset {} exhaust, repeat...".format(bmt.rank(), self.tasks[idx].dataset_name))
                    # self.tasks[idx] = SegmentedDataset(
                    # self.tasks[idx].cfg, self.tokenizer, self.max_length, transform_func=self.tasks[idx].transform_func, nthreads=self.nthreads, prefetch_slice=self.prefetch_slice
                    # )
                    # self.tasks[idx].avg_tokens_update(_runtime_ave)
                    self.tasks[idx].reset()
                else:
                    print("Rank {}, dataset {} exhaust, not repeat.".format(bmt.rank(), self.tasks[idx].dataset_name))
                    self.tasks[idx].exhaust = True
                    self.remain -= 1
                continue
            step += 1
            return dict(
                task_id=idx,
                input=data[0],
                target=data[1],
                index=data[2],
                is_long=self.tasks[idx].cfg.get("is_long", False),
            )


class UnpadBatchedMixedDataset(torch.utils.data.IterableDataset):
    def __init__(self, mixed_dataset, batch_size, max_length, pose_prob=0.0, pose_scaling_factor=1.0, compact=False):
        self.max_total_length = batch_size * max_length
        self.batch_size = 1
        # setting compact=True concats segments orignated from the same input
        # into a long sequence. the relative order of segments should be preserved
        # in mixed_dataset, e.g.,
        # - ok: task1_seg1, task2_seg1, task1_seg2, task1_seg3
        # - not_ok:  task1_seg1, task1_seg3, task2_seg1, task1_seg2
        self.compact = compact

        self.total_length = 0
        self.task2seqs = defaultdict(list)
        self.mixed_dataset = mixed_dataset
        self._max_length = max_length
        self._pose_prob = pose_prob
        self._pose_scaling_factor = pose_scaling_factor
        if self._pose_prob > 0.0:
            self._scaled_max_length = int(self.max_total_length * self._pose_scaling_factor)
        else:
            self._scaled_max_length = max_length

    def put(self, sample):
        self.total_length += len(sample["target"])
        task_id = sample["task_id"]
        if self.compact and self.task2seqs[task_id]:
            last = self.task2seqs[task_id][-1]
            if last["target"][-1] != self.mixed_dataset.eos_token_id:
                # concatenate sequantial segments for longer context modeling: why not?
                last["input"].extend(sample["input"])
                last["target"].extend(sample["target"])
                return
        self.task2seqs[task_id].append(sample)

    def _pose_preprocess(
        self,
        input_ids: NDArray[np.int32],
    ) -> NDArray[np.int32]:
        """[PoSE](https://arxiv.org/abs/2309.10400v2)
        GitHub implementation: https://github.com/dwzhu-pku/PoSE/blob/master/src/train_pose.py#L156
        """
        len_chunk = min(len(input_ids), self._max_length)
        len_input = len(input_ids)
        # Chunk input randomly to fit max_length if needed
        lt1 = 0
        rt1 = random.randint(0, (len_chunk + 1) // 2)  # Fist chunk only contains 1/2 tokens at most
        rt2 = random.randint(lt1 + len_chunk, len_input)  # Second chunk can randomly shift when not filled max_length
        lt2 = rt2 - (len_chunk - (rt1 - lt1))  # assure all tokens are used
        chunked_input_ids = np.concatenate([input_ids[lt1:rt1], input_ids[lt2:rt2]], axis=-1)
        # Generate PoSE position ids
        position_ids = np.arange(len(chunked_input_ids), dtype=np.int32)
        len_position_ids = len(position_ids)
        lt = 0
        rt = random.randint(lt, self._scaled_max_length - len_position_ids)
        position_ids[: rt1 - lt1] += lt
        position_ids[rt1 - lt1 :] += rt
        return position_ids

    def pop(self):
        indexes = defaultdict(list)
        lengths = []

        inputs = torch.zeros((self.batch_size, self.max_total_length), dtype=torch.int32)
        targets = torch.full((self.batch_size, self.max_total_length), dtype=torch.int32, fill_value=IGNORE_TGT)
        task_ids = torch.full((self.batch_size, self.max_total_length), dtype=torch.int32, fill_value=-1)
        position_ids = torch.zeros((self.batch_size, self.max_total_length), dtype=torch.int32)

        span_begin = 0
        for samples in self.task2seqs.values():
            while samples:
                sample = samples.pop()
                span_end = span_begin + len(sample["input"])
                inputs[0, span_begin:span_end] = torch.tensor(sample["input"], dtype=torch.int32)
                targets[0, span_begin:span_end] = torch.tensor(sample["target"], dtype=torch.int32)
                task_ids[0, span_begin:span_end] = torch.tensor(sample["task_id"], dtype=torch.int32)
                if not sample["is_long"] and self._pose_prob > 0.0 and random.uniform(0, 1) < self._pose_prob:
                    _span_position_ids = self._pose_preprocess(sample["input"])
                else:
                    _span_position_ids = np.arange(len(sample["input"]), dtype=np.int32)
                position_ids[0, span_begin:span_end] = torch.from_numpy(_span_position_ids)
                # position_ids[0, span_begin:span_end] = torch.arange(len(sample["input"]), dtype=torch.int32)
                lengths.append(len(sample["target"]))
                indexes[int(sample["task_id"])].append(sample["index"])
                self.total_length -= len(sample["target"])
                span_begin = span_end

        cu_seqlens = torch.cat(
            [torch.tensor([0] + lengths).cumsum(dim=-1), torch.tensor([self.max_total_length], dtype=torch.int32)],
            dim=0,
        ).int()
        batch = {
            "inputs": inputs,
            "targets": targets,
            "task_ids": task_ids,
            "indexes": indexes,
            # adhere to flash attention interface
            "cu_seqlens": cu_seqlens,
            "max_seqlen": int(torch.max(cu_seqlens[1:] - cu_seqlens[:-1])),
            "lengths": torch.tensor(sum(lengths)).int(),
            "task_names": self.mixed_dataset.names,
            "position_ids": position_ids,
        }
        return batch

    def will_be_full(self, sample):
        return self.total_length + len(sample["target"]) > self.max_total_length

    def __iter__(self):
        for sample in self.mixed_dataset:
            if self.will_be_full(sample):
                yield self.pop()
            self.put(sample)


class CudaPrefetcher(Iterable):
    """
    Wrap around a batch iterator for asynchornously copying data to gpu to shield memcpy latency.
    """

    def __init__(self, loader, tp_size=1, tp_rank=0):
        self.loader = iter(loader)
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            if self.tp_size > 1:
                if self.tp_rank == 0:
                    data = next(self.loader)
                    print("Rank {}, Preload data done.".format(bmt.rank()))
                    d = {}
                    with open(f"/dev/shm/BMT_TP_{bmt.config['topology'].tp_idx}.bin", "wb") as fb:
                        for key in data.keys():
                            if isinstance(data[key], torch.Tensor):
                                np_cur_data = data[key].cpu().numpy()
                                bs = np_cur_data.tobytes()
                                fb.write(bs)
                                d[key] = ["TORCH", str(np_cur_data.dtype), len(bs)] + list(np_cur_data.shape)
                            elif isinstance(data[key], np.ndarray):
                                bs = data[key].tobytes()
                                fb.write(bs)
                                d[key] = ["NUMPY", str(data[key].dtype), len(bs)] + list(data[key].shape)
                            else:
                                d[key] = data[key]
                        try:
                            _ = json.dumps(d)
                        except TypeError:
                            print(d)
                        with open(f"/dev/shm/BMT_TP_{bmt.config['topology'].tp_idx}.json", "w") as f:
                            json.dump(d, f)
                bmt.synchronize()
                if self.tp_rank != 0:
                    with open(f"/dev/shm/BMT_TP_{bmt.config['topology'].tp_idx}.json", "r") as f:
                        data = json.load(f)
                    with open(f"/dev/shm/BMT_TP_{bmt.config['topology'].tp_idx}.bin", "rb") as fb:
                        bs = fb.read()
                        offset = 0
                        for key in data.keys():
                            if isinstance(data[key], list) and len(data[key]) > 1 and data[key][0] == "NUMPY":
                                nw_offset = offset + data[key][2]
                                data[key] = np.frombuffer(bs[offset:nw_offset], dtype=data[key][1]).reshape(
                                    data[key][3:]
                                )
                                offset = nw_offset
                            elif isinstance(data[key], list) and len(data[key]) > 1 and data[key][0] == "TORCH":
                                nw_offset = offset + data[key][2]
                                data[key] = torch.from_numpy(
                                    np.frombuffer(bs[offset:nw_offset], dtype=data[key][1])
                                    .reshape(data[key][3:])
                                    .copy()
                                )
                                offset = nw_offset
                self.data = data
            else:
                self.data = next(self.loader)
        except StopIteration:
            self.data = None
            return
        with torch.cuda.stream(self.stream):
            for key in self.data.keys():
                if isinstance(self.data[key], torch.Tensor):
                    self.data[key] = self.data[key].cuda(non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = copy.deepcopy(self.data)
        self.preload()
        return data

    def __iter__(self):
        return self
