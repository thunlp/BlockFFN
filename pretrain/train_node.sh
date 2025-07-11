#!/bin/bash
set -ex

mkdir -p logs

export RUN_NODE=$1
export MASTER_ADDR=$2
export WORLD_SIZE=$3
export MASTER_PORT=$4

ROUTER_NORM_INIT_VAR=1.0 \
ROUTER_NORM_TYPE="rms" \
ROUTER_NORM_INIT_VAR=1.0 \
ROUTER_NORM_TYPE="rms" \
bash apps/dragonfly_2b/pretrain_dragonfly.sh --model_unique scaling_mla_12b_ffn_block_silu_rms_56e_trans_2e3_chunk_64_5e2_sche04_min1025 --config 1.2b_mla_block --router_activate_fn relu --ffn_activate_fn silu --ffn_type block_linear --norm_after_router rms --norm_scale 1.0 --balance_loss_coef 0 --l1_lambda 0 --router_entropy_coef 0 --transfer_lambda 2e-3 --token_balance_factor -1 --sigmoid_steep 100.0 --use_value_scheduler True --scheduler_max_factor 4.0 --scheduler_min_factor 1.025 --dataset_config sample_dataset --chunk_regularization_factor 5e-2 --chunk_regularization_length 64 --flash cuda --dataloader indexed --runtime_eval False --batch_size 12 --max_length 4096 --only_run_dataloader 0 --save_iters 1000 --lr 0.01 --router_lr 0.01 --train_iters 40000 --lr_scheduler warmupstableexp --drop_iters 560 --drop_begin 100000 --warmup_iters 0.01 --dataloader_num_threads 1 --dataloader_prefetch 2 --dataloader_prefetch_factor 50 --exp_group mla --tokenizer_path /home/test/test06/tokenizers/120k_v2.model --tensorboard_all_tasks 1 --ignore_cuda_oom 1 --stop_when_end 1 --lr_decoupled_decay False --inspect_training_dynamics 0 --force_restart True --load_grad True --grad_ckpt_num 64 --load_dataloader_ckpt 1 --load_ckpt_strict 1 --load_value_scheduler True --accu_iters 1 --router_accu_iters 1 --former_accu_iters 1 --local True --seed 2345 > logs/$1_$4.log
