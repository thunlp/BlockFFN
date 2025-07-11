#!/bin/bash

cd apps/dragonfire
export CUDA_LAUNCH_BLOCKING=1

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
args["load_dataloader_ckpt"]="True"
args["drop_begin"]="-1"
args["drop_rate"]="0.5"

args["seed"]="1234"
args["force_restart"]="False"
args["accu_iters"]="1"
args["former_accu_iters"]="-1"
args["router_aux_loss_coef"]="1e-2"
args["router_ent_loss_coef"]="0"
args["moe_top_k"]="-1"
args["moe_top_p"]="-1"
args["moe_routing_strategy"]=""
args["stat_act_window_sizes"]="8"
args["l1_reg_coef_init"]="1e-8"
args["l1_reg_coef_multiplier"]="1.2"
args["l1_reg_coef_resume"]="-1"
args["num_shared_experts"]="0"
args["activate_fn"]="silu"

args["valid_only"]="False"
args["valid_dataset"]="0322_validation"
args["valid_interval"]="-1"
args["valid_iters"]="1000"

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
    pip install bmtrain-zh==0.2.3.dev49
    pip install markdownify
    # pip install flash-attn==2.3.3
    pip install /home/songchenyang/flash_attn-2.5.6+cu118torch1.13cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    pip install h5py
    pip install transformers==4.35.0
    pip install accelerate==0.24
    pip install flask
    pip install jsonify
    pip install tenacity
fi



GPUS_PER_NODE=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
# GPUS_PER_NODE=1
echo "Using ${GPUS_PER_NODE} GPU each machine"


MODEL_UNIQUE=${args["model_unique"]}
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
OPTS+=" --router_aux_loss_coef ${args["router_aux_loss_coef"]}"
OPTS+=" --router_ent_loss_coef ${args["router_ent_loss_coef"]}"
OPTS+=" --moe_top_k ${args["moe_top_k"]}"
OPTS+=" --moe_top_p ${args["moe_top_p"]}"
OPTS+=" --moe_routing_strategy ${args["moe_routing_strategy"]}"
OPTS+=" --stat_act_window_sizes ${args["stat_act_window_sizes"]}"
OPTS+=" --l1_reg_coef_init ${args["l1_reg_coef_init"]}"
OPTS+=" --l1_reg_coef_multiplier ${args["l1_reg_coef_multiplier"]}"
OPTS+=" --l1_reg_coef_resume ${args["l1_reg_coef_resume"]}"
OPTS+=" --num_shared_experts ${args["num_shared_experts"]}"
OPTS+=" --activate_fn ${args["activate_fn"]}"
OPTS+=" --valid_interval ${args["valid_interval"]}"
OPTS+=" --valid_iters ${args["valid_iters"]}"
if [[ ${args["load_grad"]} == "True" ]]; then
    OPTS+=" --load-grad"
    OPTS+=" --grad-ckpt-num ${args["grad_ckpt_num"]}"
fi
if [[ ${args["force_restart"]} == "True" ]]; then
    OPTS+=" --force-restart"
fi
if [[ ${args["valid_only"]} == "True" ]]; then
    OPTS+=" --valid_only"
fi

if [[ ${args["async_save"]} == "True" ]]; then
    OPTS+=" --async_save"
fi

if [[ ${args["load_dataloader_ckpt"]} == "True" ]]; then
    OPTS+=" --load_dataloader_ckpt"
fi

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


mkdir -p ../../runtime/checkpoints/logs/${MODEL_UNIQUE}
OPTS+=" --log-dir ../../runtime/checkpoints/logs/${MODEL_UNIQUE}"
OPTS+=" --tensorboard ../../runtime/tensorboard/moe/${args["exp_group"]}${MODEL_UNIQUE}/"



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


filename="pretrain_dragonfly"

if [[ ${args["local"]} == "True" ]]; then
    PRETRAIN_ENTRY="$filename.py"
else
    PRETRAIN_ENTRY="$filename.py"
fi

CMD="torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${PRETRAIN_ENTRY} ${OPTS}"

echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"

$CMD
