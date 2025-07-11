#!/bin/bash

cd apps/dragonfly_2b

declare -A args  # Declare an associative array to store arguments and values

args["model_unique"]=""
args["resume_ckpt"]=""
args["config"]=""
args["flash"]="cuda"
args["batch_size"]=""
args["max_length"]="8192"
args["save_iters"]="500"
args["train_iters"]=""
args["dataset_config"]=""
args["local"]="False"
args["dataloader"]="indexed"
args["save"]="True"
args["dataloader_num_threads"]=1
args["dataloader_prefetch"]=2
args["dataloader_prefetch_factor"]=50
args["dataloader_num_workers"]=2
args["lr"]="1e-2"
args["warmup_iters"]="0.1"
args["drop_iters"]="0.1"
args["tokenizer_path"]="" #  /user/tc_agi/klara/baichuan2/baichuan2.tokenizer.model
args["is_pretrained_tokenizer"]="False"
args["load_grad"]="False"
args["grad_ckpt_num"]="160"
args["exp_group"]=""
args["ignore_cuda_oom"]="1"
args["tensorboard_all_tasks"]="1"
args["stop_when_end"]="0"
args["only_run_dataloader"]="0"
args["eps"]="1e-6"
args["inspect_iters"]="100"
args["strict_state_dict"]="1"
args["only_load_model"]="0"
args["lr_scheduler"]="warmupstabledrop"
args["resume_no_optimze"]="0"
args["tp_size"]="1"
args["parallel_load_datastate"]="128"
args["async_save"]="True"
args["load_dataloader_ckpt"]="1"
args["drop_begin"]="-1"
args["drop_rate"]="0.5"
args["use_checkpoint"]="1"
args["valid_only"]="False"
args["valid_dataset"]="0322_validation"
args["valid_interval"]="-1"
args["valid_iters"]="1000"

args["seed"]="1234"
args["force_restart"]="False"
args["accu_iters"]="1"
args["former_accu_iters"]="-1"

args["router_lr"]="-1"
args["router_accu_iters"]="-1"

args["l1_lambda_option"]="fixed"
args["l1_lambda"]="-1"
args["end_l1_lambda"]="-1"
args["start_cosine_step"]="0"
args["end_cosine_step"]="0"
args["inference"]="False"
args["task_act_stat"]="False"
args["router_activate_fn"]=""
args["ffn_activate_fn"]=""
args["ffn_type"]=""
args["balance_loss_coef"]="0"
args["router_entropy_coef"]="1e-3"

args["record_activations"]="False"
args["record_attention"]="False"
args["entry_num"]="400000"
args["mlp_remain_ratio"]="-1"
args["record_path"]="TODO"

args["moefication"]="False"
args["sample_num_per_layer"]="100000"
args["activation_dir"]="tmp"
args["norm_after_router"]="sum"
args["norm_scale"]="1.0"
args["use_value_scheduler"]="False"
args["scheduler_max_factor"]="2.0"
args["scheduler_min_factor"]="1.0"
args["load_value_scheduler"]="False"

args["stat_act_window_sizes"]="1,2,4,8,16,32,64"
args["transfer_lambda"]="-1"
args["token_balance_factor"]="-1"
args["sigmoid_steep"]="100"
args["chunk_regularization_factor"]="-1"
args["chunk_regularization_length"]="64"

# Loop through the arguments
for ((i=1; i<=$#; i++)); do
    arg="${!i}"
    # Check if the argument starts with "--"
    if [[ "$arg" == --* ]]; then
        arg_name="${arg:2}"  # Remove leading "--"
        valueid=$((i+1))
        # Get the value of the argument if it exists
        if ((i+1 <= $#)); then
            args["$arg_name"]="${!valueid}"
            i=$((i+1))  # Skip the next argument (its value)
        else
            args["$arg_name"]=""  # Set empty value if no value provided
        fi
    fi
done

# 用cmd arg 再更新一次
# Loop through the arguments
for ((i=1; i<=$#; i++)); do
    arg="${!i}"
    # Check if the argument starts with "--"
    if [[ "$arg" == --* ]]; then
        arg_name="${arg:2}"  # Remove leading "--"
        valueid=$((i+1))

        # Get the value of the argument if it exists
        if ((i+1 <= $#)); then
            args["$arg_name"]="${!valueid}"
            i=$((i+1))  # Skip the next argument (its value)
        else
            args["$arg_name"]=""  # Set empty value if no value provided
        fi
    fi
done

# Print the values of the arguments
echo "----------- CMD args ----------"
for key in "${!args[@]}"; do
    echo "$key: ${args[$key]}"
done
echo "--------- END CMD args --------"


if [[ ${args["flash"]} == "triton" ]]; then
    sudo cp /usr/local/cuda-11.6/compat/libcuda.so.510.108.03 /usr/lib/x86_64-linux-gnu/libcuda.so.510.108.03
    sudo ln /usr/lib/x86_64-linux-gnu/libcuda.so.510.108.03 /usr/lib/x86_64-linux-gnu/libcuda.so
    echo "triton flash"
fi

if [ "${args["local"]}" == "False" ]; then
    pip install msgspec
    pip install tensorboardX
    pip install tiktoken
    pip install bmtrain-zh==0.2.3.dev48
    pip install markdownify
    pip install flash-attn
    pip install h5py
    pip install transformers==4.35.0
    pip install accelerate==0.24
    pip install flask
    pip install jsonify
    pip install tenacity
    pip install sentencepiece
    pip install matplotlib
fi

GPUS_PER_NODE=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
# GPUS_PER_NODE=1
echo "Using ${GPUS_PER_NODE} GPU each machine"


MODEL_UNIQUE=${args["model_unique"]} # 
echo "model_unique: "$MODEL_UNIQUE

# --------------- 运行参数 ---------------

OPTS+=" --model-config model_configs/"${args['config']}".json" # [CHANGE]
OPTS+=" --batch-size ${args["batch_size"]}"
OPTS+=" --train-iters ${args["train_iters"]}"
OPTS+=" --save-iters ${args["save_iters"]}"
OPTS+=" --save-name cpm_live_checkpoint"
OPTS+=" --max-length ${args["max_length"]}"
OPTS+=" --lr ${args["lr"]}"
OPTS+=" --inspect-iters ${args["inspect_iters"]}"
OPTS+=" --warmup-iters ${args["warmup_iters"]}"
OPTS+=" --drop-iters ${args["drop_iters"]}"
OPTS+=" --lr_scheduler ${args["lr_scheduler"]}"
OPTS+=" --offload"
OPTS+=" --flash ${args["flash"]}"
OPTS+=" --tensorboard_all_tasks ${args["tensorboard_all_tasks"]}"
OPTS+=" --ignore_cuda_oom ${args["ignore_cuda_oom"]}"
OPTS+=" --stop_when_end ${args["stop_when_end"]}"
OPTS+=" --only_run_dataloader ${args["only_run_dataloader"]}"
OPTS+=" --eps ${args["eps"]}"
OPTS+=" --strict_state_dict ${args["strict_state_dict"]}"
OPTS+=" --only_load_model ${args["only_load_model"]}"
OPTS+=" --resume_no_optimze ${args["resume_no_optimze"]}"
OPTS+=" --tokenizer_path ${args["tokenizer_path"]}"
if [[ ${args["is_pretrained_tokenizer"]} == "True" ]]; then
    OPTS+=" --is_pretrained_tokenizer"
fi
OPTS+=" --weight-decay 0.1"
OPTS+=" --tp-size ${args["tp_size"]}"
OPTS+=" --parallel_load_datastate ${args["parallel_load_datastate"]}"
OPTS+=" --drop_begin ${args["drop_begin"]}"
OPTS+=" --drop_rate ${args["drop_rate"]}"

OPTS+=" --seed ${args["seed"]}"
OPTS+=" --grad-accum ${args["accu_iters"]}"
OPTS+=" --former-grad-accum ${args["former_accu_iters"]}"
OPTS+=" --l1_lambda_option ${args["l1_lambda_option"]}"
OPTS+=" --l1_lambda ${args["l1_lambda"]}"
OPTS+=" --end_l1_lambda ${args["end_l1_lambda"]}"
OPTS+=" --start_cosine_step ${args["start_cosine_step"]}"
OPTS+=" --end_cosine_step ${args["end_cosine_step"]}"
OPTS+=" --router_activate_fn ${args["router_activate_fn"]}"
OPTS+=" --ffn_activate_fn ${args["ffn_activate_fn"]}"
if [ "${args["ffn_type"]}" != "" ]; then
    OPTS+=" --ffn_type ${args["ffn_type"]}"
fi
OPTS+=" --balance_loss_coef ${args["balance_loss_coef"]}"
OPTS+=" --router_entropy_coef ${args["router_entropy_coef"]}"
OPTS+=" --stat_act_window_sizes ${args["stat_act_window_sizes"]}"
OPTS+=" --transfer_lambda ${args["transfer_lambda"]}"
OPTS+=" --token_balance_factor ${args["token_balance_factor"]}"
OPTS+=" --sigmoid_steep ${args["sigmoid_steep"]}"
OPTS+=" --entry_num ${args["entry_num"]}"
OPTS+=" --mlp_remain_ratio ${args["mlp_remain_ratio"]}"
OPTS+=" --record_path ${args["record_path"]}"

OPTS+=" --sample_num_per_layer ${args["sample_num_per_layer"]}"
OPTS+=" --activation_dir ${args["activation_dir"]}"

OPTS+=" --router_lr ${args["router_lr"]}"
OPTS+=" --router_accu_iters ${args["router_accu_iters"]}"

if [[ ${args["force_restart"]} == "True" ]]; then
    OPTS+=" --force-restart"
fi
if [[ ${args["inference"]} == "True" ]]; then
    OPTS+=" --inference"
fi
if [[ ${args["task_act_stat"]} == "True" ]]; then
    OPTS+=" --task_act_stat"
fi
OPTS+=" --load_dataloader_ckpt ${args["load_dataloader_ckpt"]}"
OPTS+=" --use_checkpoint ${args["use_checkpoint"]}"
OPTS+=" --valid_interval ${args["valid_interval"]}"
OPTS+=" --valid_iters ${args["valid_iters"]}"

if [[ ${args["load_grad"]} == "True" ]]; then
    OPTS+=" --load-grad"
    OPTS+=" --grad-ckpt-num ${args["grad_ckpt_num"]}"
fi
if [[ ${args["async_save"]} == "True" ]]; then
    OPTS+=" --async_save"
fi
if [[ ${args["record_attention"]} == "True" ]]; then
    OPTS+=" --record_attention"
fi
if [[ ${args["valid_only"]} == "True" ]]; then
    OPTS+=" --valid_only"
fi
OPTS+=" --norm_after_router ${args["norm_after_router"]}"
OPTS+=" --norm_scale ${args["norm_scale"]}"
OPTS+=" --scheduler_max_factor ${args["scheduler_max_factor"]}"
OPTS+=" --scheduler_min_factor ${args["scheduler_min_factor"]}"
if [[ ${args["use_value_scheduler"]} == "True" ]]; then
    OPTS+=" --use_value_scheduler"
fi
if [[ ${args["load_value_scheduler"]} == "True" ]]; then
    OPTS+=" --load_value_scheduler"
fi
OPTS+=" --chunk_regularization_factor ${args["chunk_regularization_factor"]}"
OPTS+=" --chunk_regularization_length ${args["chunk_regularization_length"]}"

if [[ ${args["dataloader"]} == "indexed" ]]; then
    OPTS+=" --dataloader_num_threads ${args["dataloader_num_threads"]}"
    OPTS+=" --dataloader_prefetch ${args["dataloader_prefetch"]}"
    OPTS+=" --dataloader_num_workers ${args["dataloader_num_workers"]}"
    OPTS+=" --dataloader_prefetch_factor ${args["dataloader_prefetch_factor"]}"
fi


# --------------- 写文件路径 ---------------
## checkpoint
if [[ ${args["save"]} == "True" ]]; then
    OPTS+=" --save ../../runtime/checkpoints/${MODEL_UNIQUE}/"
    OPTS+=" --save-model /not_exist/${MODEL_UNIQUE}/"
else
    echo "won't save model"
fi


mkdir -p ../../runtime/logs/${MODEL_UNIQUE}
OPTS+=" --log-dir ../../runtime/checkpoints/logs/${MODEL_UNIQUE}"
OPTS+=" --tensorboard ../../runtime/tensorboard/${args["exp_group"]}${MODEL_UNIQUE}/"



if [[ ${args["local"]} == "True" ]]; then
    current_dir=$(pwd)
    OPTS+=" --dataset ${current_dir}/dataset_configs/${args["dataset_config"]}.json"
    OPTS+=" --valid_dataset ${current_dir}/dataset_configs/${args["valid_dataset"]}.json"
else
    current_dir=$(pwd)
    OPTS+=" --dataset ${current_dir}/dataset_configs/${args["dataset_config"]}.json"
    OPTS+=" --valid_dataset ${current_dir}/dataset_configs/${args["valid_dataset"]}.json"
    echo "Platform config:"${PLATFORM_CONFIG_PATH}
fi


## checkpoint，兼容 CHECKPOINT 和 LATEST_CHECKPOINT。debug 时建议不加载 checkpoint，启动会比较快
if [ "${args["resume_ckpt"]}" != "" ]; then
  OPTS+=" --load ${args["resume_ckpt"]}"
  echo "!!! Resume from ${args["resume_ckpt"]}"
else
  echo "No checkpoint to load"
fi


if [[ ${args["record_activations"]} == "True" ]]; then
    PRETRAIN_ENTRY="record_activations.py"
elif [[ ${args["moefication"]} == "True" ]]; then
    PRETRAIN_ENTRY="moefication_dragonfly.py"
else
    PRETRAIN_ENTRY="pretrain_dragonfly.py"
fi


CMD="torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${PRETRAIN_ENTRY} ${OPTS}"

echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"

$CMD
