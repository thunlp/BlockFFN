#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2023 AI, ZHIHU Inc. (zhihu.com)
#
# @author: shengdinghu
# @date: 2024/01/08
import argparse
import os

import h5py


def build_index_from_jsonl(path):
    data_path = os.path.join(path, "data.jsonl")
    assert os.path.exists(data_path), f"Jsonline dataset '{data_path}' not found. It must be named as data.jsonl"
    offset = 0
    bounds = [offset]
    with open(data_path, "rb") as fin:
        for line in fin:
            offset += len(line)
            bounds.append(offset)

    with h5py.File(os.path.join(path, "index.h5"), "w") as hf:
        hf.create_dataset("index", data=bounds)

    with open(os.path.join(path, "index"), "w") as fout_txt:
        for s in bounds:
            fout_txt.write(f"{s}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", required=True, help="Path to the data.jsonl (without `data.jsonl`)")
    args = parser.parse_args()

    # before
    # /path
    #   data.jsonl

    build_index_from_jsonl(path=args.path)
    # after
    # /path
    #   data.jsonl
    #   index
    #   index.h5

    # Caution! Do not add the jsonl to git !
