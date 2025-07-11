# BlockFFN

Source codes for pre-training the baselines in paper: *BlockFFN: Towards End-Side Acceleration-Friendly Mixture-of-Experts with Chunk-Level Activation Sparsity*.

Links: [[Paper](TODO)] [[Models](https://huggingface.co/SparseLLM)]

### Environment

Our experiment environment is given by `bmtrain.yaml`. As it may be redundant, you can also set up the environment by installing torch (with CUDA), transformers, flash-attn, accelerate, [bmtrain](https://github.com/OpenBMB/BMTrain/), and dm-tree.

### Data Preparation

1. To run these codes, you need to prepare datasets in the `jsonline` format. An example is given by `dataset/sample_dataset/data.jsonl`.
2. Then, run the following script to create the index for this dataset:
```bash
python apps/utils/data_prepare/json_to_index.py --path dataset/sample_dataset/
```
3. Prepare another transformation script to specify how to process each line into a Dict object like `{"input": "...", "output": "..."}`. An example is given by ` apps/dragonfly_2b/dataset_configs/transforms/script.py`.
4. Finally, write a dataset configuration file like `apps/dragonfly_2b/dataset_configs/sample_dataset.json`. If you have multiple datasets, you should specify the mixture ratio using the `abs_weight` field. Besides, remember to set the `path` field to the directory containing `data.jsonl` and set the `transforms` field to the transformation script.

### Pre-Training

In `entrance.sh`, we provide 16 scripts to pre-train baselines, including 4 baseline MoE architectures (i.e., TopK, DSMoE, GRIN, and ReMoE) of 4 scales: Small (0.1B), Medium (0.5B), Large (0.8B), and XLarge (1.2B). You can copy the corresponding scripts into `train.sh` and then launch pre-training with the following command:
```bash
bash train_node.sh <current_address> <host_address> <number_of_nodes> <port_number>
```
This command also works for distributed training with multiple nodes. For the simplest one-node training, the command can just be like: `bash train_node.sh localhost localhost 1 1234`.
