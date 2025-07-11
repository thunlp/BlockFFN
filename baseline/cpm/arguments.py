# coding=utf-8
# Copyright 2020 The OpenBMB team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse


def add_model_config_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group("model", "model configuration")
    group.add_argument("--model-config", type=str, help="model configuration file")
    group.add_argument("--vocab", type=str, default=None, help="model vocabulary file")
    group.add_argument("--eps", type=float, default=1e-5, help="eps in layernorm")
    # group.add_argument("--qk_norm", action="store_true", default=False, help="qk layernorm")
    return parser


def add_training_args(parser: argparse.ArgumentParser):
    """Training arguments."""

    group = parser.add_argument_group("train", "training configurations")
    group.add_argument("--platform-config", type=str, default="platform_config.json", help="Path to platform config")
    group.add_argument("--dataset", type=str, default="dataset.json", help="Path to dataset")
    group.add_argument("--val-dataset", type=str, default="dataset.json", help="Path to val dataset")
    group.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to a directory containing a model checkpoint.",
    )

    group.add_argument(
        "--load-grad",
        action="store_true",
        default=False,
        help="Load the gradient states",
    )

    group.add_argument(
        "--grad-ckpt-num",
        type=int,
        default=0,
        help="grad file num (only work when --load-grad from files less than world-size )",
    )

    group.add_argument(
        "--load-start-step",
        action="store_true",
        default=False,
        help="Load the step state from checkpoints",
    )

    group.add_argument(
        "--save",
        type=str,
        default=None,
        help="Output directory to save checkpoints to.",
    )
    group.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Output filename to save checkpoints to.",
    )
    group.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Output directory to save model to.",
    )

    group.add_argument(
        "--tensorboard",
        type=str,
        default=None,
        help="tensorboard directory",
    )
    group.add_argument("--force-restart", action="store_true",
                       help="force to restart from step 0 (all elements)")

    group.add_argument("--inspect-iters", type=int, default=1000, help="number of inspecting")
    group.add_argument("--batch-size", type=int, default=32, help="Data Loader batch size")
    group.add_argument("--num-micro-batches", type=int, default=16)
    group.add_argument("--clip-grad", type=float, default=1.0, help="gradient clipping")
    group.add_argument("--grad-accum", type=int, default=1, help="gradient accum steps")
    group.add_argument("--former-grad-accum", type=int, default=-1, help="gradient accum steps")
    group.add_argument(
        "--train-iters",
        type=int,
        default=1000000,
        help="total number of iterations to train over all training runs",
    )
    group.add_argument("--max-length", type=int, default=512, help="max length of input")
    group.add_argument("--min-length", type=int, default=None, help="only for speed test")

    group.add_argument("--seed", type=int, default=1234, help="random seed for reproducibility")

    # Learning rate.
    group.add_argument("--lr", type=float, default=1.0e-4, help="initial learning rate")
    group.add_argument("--lr_scheduler", type=str, default="cosine", help=" learning rate scheduler")

    group.add_argument("--weight-decay", type=float, default=1.0e-2, help="weight decay rate")
    group.add_argument("--loss-scale", type=float, default=65536, help="loss scale")
    group.add_argument("--max-loss-scale", type=float, default=float("inf"), help="loss scale")
    group.add_argument("--min-loss-scale", type=float, default=1, help="loss scale")
    group.add_argument("--loss-scale-steps", type=float, default=1024, help="loss scale")

    group.add_argument(
        "--warmup-iters",
        type=float,
        default=0.01,
        help="percentage of data to warmup on (.01 = 1% of all " "training iters). Default 0.01",
    )
    group.add_argument(
        "--drop-iters",
        type=float,
        default=0.01,
        help="percentage of data to warmup on (.01 = 1% of all " "training iters). Default 0.01",
    )

    group.add_argument("--lr-decay-iters", type=int, default=None, help="lr decay steps")
    group.add_argument("--lr_end_restart", type=int, default=0, help="whether restart lr scheduler")
    group.add_argument("--start-step", type=int, default=0, help="step to start or continue training")
    group.add_argument("--concat-data", action="store_true", help="whether we concatenate the dialogues")
    group.add_argument("--offload", action="store_true", help="whether we use offload_adam")
    group.add_argument("--new-bmt", action="store_true", help="new bmt without ckpt")
    group.add_argument("--flash", default="none", choices=["none", "1d", "triton", "cuda"])
    group.add_argument("--use-jfs-data", action="store_true", help="whether we use juicefs dataset")
    group.add_argument("--tp-size", default=1, type=int)
    group.add_argument("--pp-size", default=1, type=int)
    group.add_argument("--bf16", action="store_true", help="whether we use bf16")
    group.add_argument("--dataloader_num_threads", default=3, type=int, help="Only useful in indexed dataest.")
    group.add_argument("--dataloader_prefetch", default=200, type=int, help="Only useful in indexed dataest.")
    group.add_argument("--dataloader_num_workers", default=4, type=int, help="Only useful in indexed dataest.")
    group.add_argument("--dataloader_prefetch_factor", default=50, type=int, help="Only useful in indexed dataest.")
    group.add_argument(
        "--dataloader",
        default="indexed",
        type=str,
        help="dataloader type, 'indexed' for indexed dataset, 'normal' for normal dataset",
    )
    group.add_argument("--stop_when_end", default=0, type=int, help="Whether to stop training when we reach end_iter")
    group.add_argument(
        "--data_len_threshold",
        default=512,
        type=int,
        help="If the average length of a sequence is less than this int, mean the sample is biased. ",
    )
    group.add_argument(
        "--only_run_dataloader", default=0, type=int, help="Whether to only run dataloader to check data. "
    )
    group.add_argument(
        "--only_load_model", default=0, type=int, help="Whether to only load a model ckpt, without anything else."
    )
    group.add_argument(
        "--resume_no_optimze",
        default=0,
        type=int,
        help="The number of steps that does not add optimization after resume",
    )
    group.add_argument(
        "--parallel_load_datastate",
        default=256,
        type=int,
        help="The number of parallel workers to load dataset state",
    )
    group.add_argument(
        "--async_save",
        action="store_true",
        help="whether to save artifacts asynchronously",
    )
    group.add_argument(
        "--load_dataloader_ckpt",
        action="store_true",
        help="whether to load dataloader checkpoint",
    )
    group.add_argument(
        "--drop_begin",
        default=-1,
        type=int,
        help="The number of steps that starts to drop lr"
    )
    group.add_argument(
        "--drop_rate",
        default=0.5,
        type=float,
        help="The number rate"
    )
    group.add_argument(
        "--valid_dataset",
        default="",
        type=str,
        help="The valid dataset path."
    )
    group.add_argument(
        "--valid_interval",
        default=-1,
        type=int,
        help="The interval of validation."
    )
    group.add_argument(
        "--valid_iters",
        default=1000,
        type=int,
        help="The maximum iteration of validation."
    )
    group.add_argument(
        "--valid_only",
        action="store_true",
        help="If set, exit after first validation."
    )

    return parser


def add_pretrain_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("pretrain", "pretrain configurations")
    group.add_argument(
        "--save-iters",
        type=int,
        default=1000,
        help="number of iterations between saves",
    )
    group.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="log directory",
    )
    group.add_argument(
        "--worker-name",
        type=str,
        default=None,
        help="worker name",
    )
    return parser


def add_tokenizer_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("tokenizer", "tokenizer configurations")
    group.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="tokenizer_path",
    )
    group.add_argument("--is_pretrained_tokenizer", action="store_true", default=False, help="use from pretrained load")
    return parser


def add_finetune_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("finetune", "finetune configurations")
    group.add_argument("--epoch", type=int, default=1, help="number of training epochs")
    group.add_argument("--task-name", type=str, default="task", help="name of training task")
    group.add_argument("--save-epochs", type=int, default=1, help="number of training epochs between saves")
    group.add_argument("--save-steps", type=int, default=0, help="number of training steps between saves")
    group.add_argument(
        "--drop-last",
        action="store_true",
        default=False,
        help="drop data from each epoch that cannot be formed into a complete batch at the end",
    )
    group.add_argument("--delta-tuning", action="store_true", default=False)
    group.add_argument("--each-epoch-save", default=False)
    group.add_argument("--train-task-id", type=int, default=-1)
    return parser


def add_rhlf_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("rhlf", "rhlf configurations")

    group.add_argument(
        "--load-reward",
        type=str,
        default=None,
        help="Path to reward model checkpoint.",
    )
    group.add_argument("--actor-lr", type=float, default=1.0e-5, help="actor learning rate")
    group.add_argument("--critic-lr", type=float, default=1.0e-6, help="critic learning rate")
    group.add_argument("--actor-loss-scale", type=float, default=65536, help="actor loss scale")
    group.add_argument("--critic-loss-scale", type=float, default=65536, help="critic loss scale")
    group.add_argument("--avg-reward-bias", type=float, default=0, help="reward bias")
    group.add_argument("--actor-delay-step", type=int, default=0, help="actor delay step")
    group.add_argument("--entropy-coef", type=float, default=-1.0, help="coef of policy entropy")
    ##
    return parser


def add_simple_rhlf_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("simple_rhlf", "simple rhlf configurations")
    group.add_argument("--epoch", type=int, default=1, help="number of training epochs")
    group.add_argument("--sample-batch-size", type=int, default=32, help="Data Loader sample batch size")
    group.add_argument("--load-reward", type=str, default=None, help="Path to reward model checkpoint")
    group.add_argument("--avg-reward-bias", type=float, default=0, help="reward bias")
    group.add_argument("--sample-min-length", type=int, default=20, help="sample-min-length")
    group.add_argument("--sample-max-inp-length", type=int, default=1024, help="sample-max-inp-length")
    group.add_argument("--sample-max-length", type=int, default=64, help="sample-max-length")
    group.add_argument("--sample-repetition-penalty", type=float, default=1.05, help="sample-repetition-penalty")
    group.add_argument("--sample-temperature", type=float, default=1.0, help="sample-temperature")
    group.add_argument("--encode-max-length", type=int, default=1024, help="encode-max-length")
    group.add_argument("--generate-max-length", type=int, default=64, help="generate-max-length")
    group.add_argument("--value-loss-weight", type=float, default=0.1, help="value-loss-weight")
    group.add_argument("--ptx-loss-weight", type=float, default=0.001, help="ptx-loss-weight")
    group.add_argument("--save-epochs", type=int, default=1, help="number of training epochs between saves")
    ##
    return parser


def add_feedback_learning_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("rrhf", "rrhf configurations")
    group.add_argument("--length-penalty", type=float, default=1.0, help="length_penalty")
    group.add_argument("--feedback-weight", type=float, default=1.0, help="feedback_weight")
    group.add_argument("--sample-num", type=int, default=6, help="sample_num")
    group.add_argument("--dpo-beta", type=float, default=1.0, help="dpo_beta")
    group.add_argument("--stable-alignment-margin", type=float, default=1.0, help="stable_alignment_margin")
    group.add_argument("--feedback-learning-type", type=str, default="RRHF", help="feedback_learning_type")
    group.add_argument("--save-iters", type=int, default=1000, help="number of iterations between saves")
    ##
    return parser


def add_model_change_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("model_change", "model change during pretraining")
    group.add_argument("--strict_state_dict", type=int, default=1, help="strict_state_dict")
    ##
    return parser


def add_log_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("log", "log configurations")
    group.add_argument("--tensorboard_all_tasks", type=int, default=0, help="log")
    return parser


def add_error_handle_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("error_handle", "error_handle configurations")
    group.add_argument(
        "--ignore_cuda_oom", type=int, default=1, help="continue training by ingore the batch that causes oom"
    )
    return parser


def add_runtime_eval_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("runtime eval args", "runtime evaluation by submitting a job")
    group.add_argument(
        "--runtime_eval",
        action="store_true",
        help="whether to use runtime_eval. Only if this is set to True, the following variables will be useful",
    )
    group.add_argument("--eval_jeeves_auth", type=str, default="", help="auth, press f12 on jeeves platform to get")
    group.add_argument("--eval_project_id", type=str, default=None, help="project id")
    group.add_argument("--eval_run_cmd", type=str, default="", help="cmd for eval")
    group.add_argument(
        "--eval_git_path",
        type=str,
        default="",
        help="git path of evaluation code",
    )
    group.add_argument("--eval_git_branch", type=str, default="master", help="git branch of evaluation code")
    group.add_argument("--eval_node_num", type=int, default=1, help="using 1 node to evaluate")
    group.add_argument("--eval_gpu_num", type=int, default=1, help="using 1 gpu per node to evaluate")
    group.add_argument("--eval_tasks_config", type=str, default="", help="evaluate tasks' config")
    group.add_argument("--eval_model_backend", default="torch", type=str, help="model_backend")
    group.add_argument("--eval_cpm_live_branch", type=str, default="master", help="dependency cpm-live branch")
    group.add_argument(
        "--eval_at_start", action="store_true", help="whether to eval at the first epoch, default to false"
    )

    return parser


def add_reward_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("reward", "reward configurations")
    group.add_argument("--load-all", type=str, default=None, help="Path to a directory containing a model checkpoint.")
    ##
    return parser


def add_long_context_extend_args(parser: argparse.ArgumentParser):
    """long context extending arguments."""
    group = parser.add_argument_group("long_context_extend", "long context extend configurations")
    group.add_argument("--pose-prob", default=0.0, type=float, help="Sample-level PoSE probability")
    group.add_argument(
        "--pose-scaling-factor",
        default=1.0,
        type=float,
        help="PoSE scaling factor, simulate input length = max_length * pose_scaling_factor",
    )
    group.add_argument(
        "--rope-scaling-type",
        default="",
        type=str,
        choices=["Linear", "NTK-aware", "Dynamic NTK", "NTK-by-parts", "YaRN", ""],
        help="Context scaling type",
    )
    group.add_argument("--rope-scaling-factor", default=1, type=int, help="Context scaling factor")
    group.add_argument(
        "--orig-max-length", default=8192, type=int, help="Original context length before context extending"
    )
    return parser


def add_moe_args(parser: argparse.ArgumentParser):
    """moe arguments."""
    group = parser.add_argument_group("moe", "configurations for mixture-of-experts structure")
    group.add_argument("--moe_top_k", default=-1, type=int, help="the number of selected experts")
    group.add_argument("--moe_top_p", default=-1, type=float, help="the threshold for selecting experts")
    group.add_argument("--moe_routing_strategy", default="", type=str, help="the MoE routing strategy")
    group.add_argument("--router_aux_loss_coef", default=1e-2, type=float, help="coefficient of aux router loss")
    group.add_argument("--router_ent_loss_coef", default=1e-4, type=float, help="coefficient of router entropy loss")
    group.add_argument("--stat_act_window_sizes", type=str, default="2,4,8,16",
                       help="the list of window sizes for activation statistics")
    group.add_argument("--l1_reg_coef_init", default=1e-8, type=float, help="initial coefficient of the l1_reg loss")
    group.add_argument("--l1_reg_coef_multiplier", default=1.2, type=float, help="multiplier of the l1_reg coefficient")
    group.add_argument("--l1_reg_coef_resume", default=-1, type=float, help="resume value of the l1_reg coefficient")
    group.add_argument("--num_shared_experts", type=int, default=0, help="number of shared experts")
    group.add_argument("--activate_fn", type=str, default="", help="the activation function")
    return parser

def get_args(
    pretrain: bool = False,
    finetune: bool = False,
    rhlf: bool = False,
    simple_rlhf: bool = False,
    feedback_learning: bool = False,
    reward: bool = False,
):
    parser = argparse.ArgumentParser()
    parser = add_model_config_args(parser)  # config file need to be exported with model/ckpt
    parser = add_training_args(parser)
    if pretrain:
        parser = add_pretrain_args(parser)
        parser = add_runtime_eval_args(parser)
        parser = add_tokenizer_args(parser)
        parser = add_log_args(parser)
        parser = add_error_handle_args(parser)
        parser = add_model_change_args(parser)
        parser = add_moe_args(parser)

    if finetune:
        parser = add_finetune_args(parser)
    if rhlf:
        parser = add_rhlf_args(parser)
    if simple_rlhf:
        parser = add_simple_rhlf_args(parser)
    if feedback_learning:
        parser = add_feedback_learning_args(parser)
    if reward:
        parser = add_reward_args(parser)
    parser = add_long_context_extend_args(parser)

    args = parser.parse_args()

    return args
