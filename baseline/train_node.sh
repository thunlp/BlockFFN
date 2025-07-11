#!/bin/bash
set -ex

mkdir -p logs

export RUN_NODE=$1
export MASTER_ADDR=$2
export WORLD_SIZE=$3
export MASTER_PORT=$4

bash apps/dragonfire/pretrain_dragonfly.sh --model_unique scaling_mla_12b_moe_10in56_relu_fix_reg1e8 --config moe_1.2b_mla_10in56 --dataset_config sample_dataset --flash cuda --dataloader indexed --runtime_eval False --batch_size 12 --max_length 4096 --router_aux_loss_coef 1e-4 --l1_reg_coef_init 1e-8 --l1_reg_coef_resume -1 --l1_reg_coef_multiplier 1.05 --router_ent_loss_coef 0 --moe_top_k 10 --num_shared_experts 2 --moe_top_p -1 --moe_routing_strategy relu --only_run_dataloader 0 --save_iters 1000 --lr 0.01 --train_iters 40000 --lr_scheduler warmupstableexp --drop_iters 560 --drop_begin 100000 --warmup_iters 0.01 --dataloader_num_threads 1 --dataloader_prefetch 2 --dataloader_prefetch_factor 50 --exp_group moe --tokenizer_path '/home/test/test06/tokenizers/120k_v2.model' --tensorboard_all_tasks 1 --ignore_cuda_oom 1 --stop_when_end 1 --lr_decoupled_decay False --inspect_training_dynamics 0 --force_restart True --load_grad True --grad_ckpt_num 8 --load_dataloader_ckpt 1 --load_ckpt_strict 1 --accu_iters 8 --former_accu_iters 8 --local True --seed 2345 > logs/$1_$4.log
