#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2023 AI, ZHIHU Inc. (zhihu.com)
#
# @author: ouzebin
# @date: 2023/09/27

"""
使用 IndexedDataset 前需按指定格式构建或者转换已有数据集
数据集文件结构：
- <dataset name>
    - data.jsonl  # jsonl 格式的数据，每一行一条样本
    - index  # 记录每一行 json 数据的起始 byte-offset

从头构建：直接使用 IndexedDatasetBuilder 这个 context manager
>>>    with IndexedDatasetBuilder("swear", overwrite=True) as builder:
>>>        for data in [{"input": f"screw it {i}", "output": f"for god's sake {i}"} for i in range(100)]:
>>>            builder.put(data)
转换：
从 CPM distributed_dataset 转换：使用 `cpm.dataset.tools.distributed_to_indexed`
$ python -m cpm.dataset.tools.distributed_to_indexed -i <原数据集文件夹> -o <新数据集文件夹>
已有 jsonl 数据：使用 `cpm.dataset.tools.jsonl_to_index` 构建 index 文件。需提前先把 jsonl 文件命名为
$ python -m cpm.dataset.tools.jsonl_to_index -p <数据集文件夹路径>
"""
import itertools
import math
import os
import queue
import random
import threading
import time

import bmtrain as bmt
import h5py
import numpy
import numpy as np
import torch

try:
    import msgspec

    json_decode = msgspec.json.decode
    json_encode = msgspec.json.encode
except ModuleNotFoundError:
    import json

    json_decode = json.loads
    json_encode = json.dumps

import torch
from torch.utils.data import Dataset

from cpm.utils.bitset import BitSet
from cpm.utils.bitset import bitset_diff

print_lock = threading.Lock()


def random_range(start, stop=None, step=None):
    """
    Generator of non-repeated random permutation with the same inteface of python
    `range`. Obtained from https://stackoverflow.com/a/53551417
    The random.shuffle(list) and random.sample(list, len(list)) require
    materialize the lists, which result in a long initalization period.
    """
    if stop is None:
        start, stop = 0, start
    if step is None:
        step = 1
    # Use a mapping to convert a standard range into the desired range.
    mapping = lambda i: (i * step) + start
    # Compute the number of numbers in this range.
    maximum = int(math.ceil((stop - start) / step))
    if maximum == 0:
        # early return with empty range
        yield from ()
        return
    # Seed range with a random integer.
    value = random.randint(0, maximum)
    # Construct an offset, multiplier, and modulus for a linear
    # congruential generator. These generators are cyclic and
    # non-repeating when they maintain the properties:
    #
    #   1) "modulus" and "offset" are relatively prime.
    #   2) ["multiplier" - 1] is divisible by all prime factors of "modulus".
    #   3) ["multiplier" - 1] is divisible by 4 if "modulus" is divisible by 4.

    # Pick a random odd-valued offset.
    offset = random.randint(0, maximum) * 2 + 1
    # Pick a multiplier 1 greater than a multiple of 4.
    multiplier = 4 * (maximum // 4) + 1
    # Pick a modulus just big enough to generate all numbers (power of 2).
    modulus = int(2 ** math.ceil(math.log2(maximum)))
    # Track how many random numbers have been returned.
    found = 0
    while found < maximum:
        # If this is a valid value, yield it in generator fashion.
        if value < maximum:
            found += 1
            yield mapping(value)
        # Calculate the next value in the sequence.
        value = (value * multiplier + offset) % modulus


class Range(object):
    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step

    def __repr__(self):
        return f"Range({self.start}, {self.stop}, {self.step})"

    def iterate(self):
        yield from range(self.start, self.stop, self.step)

    def list(self):
        return list(range(self.start, self.stop, self.step))

    def subrange(self, split, nsplits):
        # strided spliting range params
        # e.g., [0, 3, 5, 7, 9] can be split into [0, 5, 9] and [3, 7]
        return Range(self.start + self.step * split, self.stop, self.step * nsplits)

    def random_iterate(self):
        yield from random_range(self.start, self.stop, self.step)


def safe_print(*args, **kargs):
    if "flush" in kargs:
        flush = kargs["flush"]
        del kargs["flush"]
    else:
        flush = True
    with print_lock:
        print(*args, **kargs, flush=flush)


def concurrent_info():
    # world_size, rank = bmt.world_size(), bmt.rank()
    world_size = bmt.config["world_size"] // bmt.config["tp_size"]
    rank = bmt.config["topology"].tp_idx
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        nworkers, worker_id = 1, 1
    else:
        nworkers, worker_id = worker_info.num_workers, worker_info.id
    # print("concurrent_info: (world_size, rank, nworkers, worker_id): {}".format((world_size, rank, nworkers, worker_id)))
    return world_size, rank, nworkers, worker_id


class IndexedDataset(Dataset):
    def __init__(self, path, max_retry=1, retry_sleep=5):
        super().__init__()
        self.path = path
        self.max_retry = max_retry
        self.retry_sleep = retry_sleep
        self.bounds = None
        self.h5file = None
        self.build_index()

    def size(self):
        return self.bounds[-1]

    def _build_index_h5(self):
        index_path = os.path.join(self.path, "index.h5")
        if os.path.getsize(index_path) > 104857600:
            self.h5file = h5py.File(os.path.join(self.path, "index.h5"), "r")
            self.bounds = self.h5file["index"]
        else:
            # only load index into memory when it is small (< 100 Mb)
            # to avoid keeping to many file handlers
            self.h5file = None
            with h5py.File(index_path, "r") as hf:
                self.bounds = np.array(hf["index"])

    def __del__(self):
        if self.h5file is not None:
            self.h5file.close()

    def build_index(self):
        s = time.time()

        txt_size = os.path.getsize(os.path.join(self.path, "index"))
        if txt_size > 0.5 * 1024**3 and os.path.exists(os.path.join(self.path, "index.h5")):
            source = "h5"
            self._build_index_h5()
        else:
            source = "txt"
            self._build_index_txt()
        e = time.time()
        bmt.print_rank("build_index_{} from {}, using {:.2f}s".format(source, self.path, e - s))

    def _build_index_txt(self):
        with open(os.path.join(self.path, "index"), "r") as fin:
            self.bounds = [int(line) for line in fin]
            self.nlines = len(self.bounds)

    def safe_read(self, i_or_s, offset, size):
        for retry in itertools.count():
            try:
                # destroy the file identifier to avoid pressure on alluxio
                # buffering=0 to avoid overhead during file.seek() and open()
                with open(os.path.join(self.path, "data.jsonl"), "rb", buffering=0) as fin:
                    fin.seek(offset)
                    raw = fin.read(size)
                return raw
            except OSError as e:
                if retry >= self.max_retry:
                    raise OSError(f"reach maximum #retry: {retry}, the file system is broken.")
                safe_print(
                    f"retry loading {self.path}:{i_or_s} in {self.retry_sleep} seconds due to error: '{repr(e)}'"
                )
                time.sleep(self.retry_sleep)
            except ValueError as e:
                # reading error during python io, skip
                safe_print(f"skipping {self.path}:{i_or_s} due to error: '{repr(e)}'")
                return None

    def __repr__(self):
        return (
            f"IndexedDataset(path={self.path}, max_retry={self.max_retry}, retry_sleep={self.retry_sleep}) "
            f"with {len(self)} entries."
        )

    def __len__(self):
        return len(self.bounds) - 1

    def bound_idx(self, key, strict=False):
        # bound index within the standard range: [0, len(self))
        # useful for tracing buggy entries
        if strict and not (-len(self) <= key < len(self)):
            raise IndexError(f"Index {key} out of range for '{self.path}'")
        key = min(max(-len(self), key), len(self))  # bound key within [-len(self), len(self)]
        key = key if key > 0 else key % len(self)  # remap negative id to positive ones
        return key

    def __getitem__(self, key):
        # supports list-like slicing and indexing. strided slicing is not currently supported.
        # ok: self[1], self[-1], self[1:3], self[-10:-5], self[-10:-5:1], self[:5]
        # not ok: self[-10:-5:2], self[:100:3]
        if isinstance(key, slice):
            if not (key.step == 1 or key.step is None):
                raise ValueError(f"slice step should be 1 or None, not {key.step}")
            start = self.bound_idx(0 if key.start is None else key.start)
            stop = max(self.bound_idx(len(self) if key.stop is None else key.stop), start)
            if stop == start:
                # early returning empty slice
                return list()
            offset, size = self.bounds[start], self.bounds[stop] - self.bounds[start]
            raw = self.safe_read(key, offset, size)
            if raw is None:
                return None
            else:
                return [
                    raw[s - offset : e - offset]
                    for s, e in zip(self.bounds[start:stop], self.bounds[start + 1 : stop + 1])
                ]

        elif isinstance(key, int):
            key = self.bound_idx(key, strict=True)
            offset, size = self.bounds[key], self.bounds[key + 1] - self.bounds[key]
            raw = self.safe_read(key, offset, size)
            return raw
        else:
            raise TypeError(f"indices must be integers or slices, not {type(key)}")


class PrefetchDecodeDataset(IndexedDataset):
    # Add prefetched sampled iterator and state_dict tracking upon the simple IndexedDataset
    # Add safe decoding in iterator
    def __init__(self, *args, decode=json_decode, allow_repeat=False, **kargs):
        super().__init__(*args, **kargs)
        self.decode = decode
        self.allow_repeat = allow_repeat

    def safe_decode(self, i, raw):
        if raw is None:
            return None
        try:
            return self.decode(raw)
        except Exception as e:
            safe_print(f"Skip decoding {self.path}:{i} due to error '{e}'")
            return None

    def __getitem__(self, key):
        raw = super().__getitem__(key)
        if raw is None:
            return None
        # key should be either a slice or an integer as checked in IndexedDataset
        if isinstance(key, slice):
            return [self.safe_decode(i, r) for i, r in zip(range(key.start, key.stop), raw)]
        else:
            return self.safe_decode(key, raw)

    def loader(self, q, lid, keys, stop, used=None):
        # concurrent prefetching worker
        if used is None:
            used = BitSet()
        try:
            for key in keys:
                if stop.is_set():
                    break
                # key is either a slice or an integer index
                index = range(key.start, key.stop) if isinstance(key, slice) else [key]
                unused = bitset_diff(set(index), used)
                if not unused:
                    # skip used slice / item
                    continue
                if not q.empty():
                    # avoid breaking the distributed file system with large io load
                    time.sleep(random.random() * 2)
                # read raw data with IndexedDataset.__getitem__, suspend decoding util we really need it
                raw = super().__getitem__(key)
                if raw is None:
                    continue
                # filter used data
                items = [(i, s) for i, s in zip(index, raw if len(index) > 1 else [raw]) if i in unused]
                random.shuffle(items)
                for item in items:
                    q.put(item)
        finally:
            # signaling the end of iteration to the main thread
            q.put(StopIteration(lid))

    def _iterate(self, key_groups, nprefetch=1000, used=None):
        # helper function for concurrent prefetching
        q = queue.Queue(maxsize=nprefetch)
        stop = threading.Event()
        alive = set()
        try:
            for lid, keys in enumerate(key_groups):
                loader = threading.Thread(target=self.loader, args=(q, lid, keys, stop, used), daemon=True)
                loader.start()
                alive.add(lid)
            while True:
                try:
                    item = q.get(block=False)
                except queue.Empty:
                    if not alive:
                        # no alive loader, thus no item will be put in the queue
                        break
                    else:
                        # new item will be put later, wait for a while
                        time.sleep(0.1)
                        continue
                if isinstance(item, StopIteration):
                    alive.remove(item.value)
                    continue
                i, raw = item
                data = self.safe_decode(i, raw)
                if data is None:
                    continue
                yield i, data
        finally:
            # ask daemon loaders to stop
            stop.set()

    def iterate(self, nthreads=3, prefetch_sample=100, used=None, process_group=None):
        world_size, rank, nworkers, worker_id = concurrent_info(process_group)
        nloaders = world_size * nworkers * nthreads
        if len(self) < nloaders:
            raise ValueError(
                f"more concurrent loaders ({nloaders}) than data entries ({len(self)}) in '{self.path}', "
                f"please constrain either "
                f"world_size={world_size}, num_workers={nworkers} or num_threads={nthreads}."
            )
        r = Range(0, len(self), 1)
        # split index among multi-gpu workers
        r = r.subrange(split=rank, nsplits=world_size)
        # split index among multi-process dataloader workers
        r = r.subrange(split=worker_id, nsplits=nworkers)
        # split index among multi-threaded loaders
        id_groups = [r.subrange(split=tid, nsplits=nthreads).random_iterate() for tid in range(nthreads)]
        return self._iterate(id_groups, nprefetch=prefetch_sample, used=used)

    def sliced_iterate(self, nthreads=1, prefetch_slice=3, slice_size=500, used=None):
        world_size, rank, nworkers, worker_id = concurrent_info()
        nloaders = world_size * nworkers * nthreads
        if len(self) < nloaders:
            if not self.allow_repeat:
                raise ValueError(
                    f"more concurrent loaders ({nloaders}) than data entries ({len(self)}) in '{self.path}', "
                    f"please constrain either "
                    f"world_size={world_size}, num_workers={nworkers} or num_threads={nthreads}."
                )
            else:
                duplicated_factor = math.ceil(nloaders / len(self))
                # In this case, slice size is 1
                r = Range(0, len(self), 1)
                # split index among grouped multi-gpu workers
                r = r.subrange(split=rank // duplicated_factor, nsplits=math.ceil(world_size / duplicated_factor))
                # # split index among multi-threaded loaders
                r = r.subrange(split=worker_id, nsplits=nworkers)
        else:
            nslices = int(math.ceil(len(self) / slice_size))

            if nslices < nloaders:
                safe_print(
                    f"fail to distribute {nslices} slices from '{self.path}' to {nloaders} concurrent loaders, "
                    f"reduce slice_size from {slice_size} to {len(self) // nloaders}."
                )
                slice_size = len(self) // nloaders

            # we only iteratre through start ids as they uniquely mark each slice
            r = Range(0, len(self), slice_size)
            # split index among multi-gpu workers
            r = r.subrange(split=rank, nsplits=world_size)
            # split index among multi-process dataloader workers
            r = r.subrange(split=worker_id, nsplits=nworkers)
            # split index among multi-threaded loaders
        slice_groups = [
            (slice(s, s + slice_size) for s in r.subrange(tid, nthreads).random_iterate()) for tid in range(nthreads)
        ]
        return self._iterate(slice_groups, nprefetch=prefetch_slice * slice_size, used=used)


class IndexedDatasetBuilder:
    def __init__(self, path, overwrite=False):
        self.path = path
        self.index_path = os.path.join(self.path, "index.h5")
        self.index_path_txt = os.path.join(self.path, "index")
        self.data_path = os.path.join(self.path, "data.jsonl")
        if not overwrite:
            assert not os.path.exists(self.data_path)
            assert not os.path.exists(self.index_path)
            assert not os.path.exists(self.index_path_txt)
        self.fout = None
        self.bounds = []
        self.offset = 0

    def __enter__(self):
        os.makedirs(self.path, exist_ok=True)
        self.fout = open(self.data_path, "wb")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.bounds.append(self.offset)
        with h5py.File(os.path.join(self.index_path), "w") as hf:
            hf.create_dataset("index", data=self.bounds)
        with open(self.index_path_txt, "w") as fout_txt:
            for s in self.bounds:
                fout_txt.write(f"{s}\n")
        self.fout.close()

    def put(self, data: dict):
        s = json_encode(data) + b"\n"
        self.bounds.append(self.offset)
        self.offset += len(s)
        self.fout.write(s)


if __name__ == "__main__":
    with IndexedDatasetBuilder("swear", overwrite=True) as builder:
        for d in [{"input": f"screw it {i}", "output": f"for god's sake {i}"} for i in range(100)]:
            builder.put(d)
    dataset = IndexedDataset("swear")
    for i in range(10):
        print(dataset[random.randint(0, len(dataset) - 1)])
