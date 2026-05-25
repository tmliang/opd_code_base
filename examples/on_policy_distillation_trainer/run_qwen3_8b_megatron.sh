#!/usr/bin/env bash
# On-policy distillation | text | vLLM rollout | Megatron training | NVIDIA GPUs
set -xeuo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---- model / data ----
student_model=Qwen/Qwen3-8B
teacher_model=Qwen/Qwen3-32B

gsm8k_train=$HOME/data/gsm8k/train.parquet
gsm8k_test=$HOME/data/gsm8k/test.parquet
math_train=$HOME/data/math/train.parquet
math_test=$HOME/data/math/test.parquet
train_files=[$gsm8k_train,$math_train]
val_files=[$gsm8k_test,$math_test]

# ---- distributed ----
nnodes=1
n_gpus_per_node=8
teacher_world_size=4

# ---- batch / sequence ----
train_batch_size=128
ppo_mini_batch_size=128
max_prompt_length=1024
max_response_length=2048
ppo_max_token_len_per_gpu=24576
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

# ---- optim ----
actor_lr=1e-6

# ---- actor (megatron) ----
actor_tp=2
actor_pp=1

# ---- rollout ----
rollout_tp=2
rollout_gpu_mem_util=0.4

# ---- teacher ----
teacher_tp=2
teacher_gpu_mem_util=0.4

# ---- distillation (loss) ----
distillation_loss_mode=forward_kl_topk
distillation_topk=64
use_policy_gradient=False
loss_max_clamp=10.0
log_prob_min_clamp=-10.0

# ---- trainer ----
project_name=verl_distill_gsm8k_math
experiment_name=qwen3_8b_from_qwen3_32b_vllm_megatron
total_epochs=15
save_freq=200
test_freq=5

python3 -m verl.trainer.main_ppo \
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$student_model" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$actor_tp \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$actor_pp \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$rollout_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_gpu_mem_util \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_model_len=$max_num_tokens \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    distillation.enabled=True \
    distillation.n_gpus_per_node=$teacher_world_size \
    distillation.nnodes=$nnodes \
    distillation.teacher_models.teacher_model.model_path="$teacher_model" \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=$teacher_tp \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=$teacher_gpu_mem_util \
    distillation.teacher_models.teacher_model.inference.max_model_len=$max_num_tokens \
    distillation.distillation_loss.loss_mode=$distillation_loss_mode \
    distillation.distillation_loss.topk=$distillation_topk \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=$use_policy_gradient \
    distillation.distillation_loss.loss_max_clamp=$loss_max_clamp \
    distillation.distillation_loss.log_prob_min_clamp=$log_prob_min_clamp \
    trainer.balance_batch=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.val_before_train=False \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs \
    "$@"
