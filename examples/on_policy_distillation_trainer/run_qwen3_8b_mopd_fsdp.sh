#!/usr/bin/env bash
# On-policy distillation | multi-teacher (gsm8k text + geo3k VL) | vLLM rollout | FSDP training | NVIDIA GPUs
set -xeuo pipefail

# ---- model / data ----
student_model=Qwen/Qwen3-VL-8B-Instruct
gsm8k_teacher_model=Qwen/Qwen3-32B
geo3k_teacher_model=Qwen/Qwen3-VL-32B-Instruct

gsm8k_train=$HOME/data/gsm8k/train.parquet
gsm8k_test=$HOME/data/gsm8k/test.parquet
geo3k_train=$HOME/data/geo3k/train.parquet
geo3k_test=$HOME/data/geo3k/test.parquet
train_files=[$gsm8k_train,$geo3k_train]
val_files=[$gsm8k_test,$geo3k_test]

# ---- distributed ----
nnodes=1
n_gpus_per_node=8

# Per-teacher replicas; total teacher GPUs = sum(num_replicas) * teacher_tp
teacher_nnodes=1
teacher_num_replicas_gsm8k=1
teacher_num_replicas_geo3k=1
teacher_tp=2
teacher_world_size=$(( (teacher_num_replicas_gsm8k + teacher_num_replicas_geo3k) * teacher_tp ))
teacher_gpu_mem_util=0.4

# ---- batch / sequence ----
train_batch_size=128
ppo_mini_batch_size=128
max_prompt_length=1024
max_response_length=2048
ppo_max_token_len_per_gpu=24576
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

# ---- optim ----
actor_lr=1e-6

# ---- rollout ----
rollout_tp=2
rollout_gpu_mem_util=0.4

# ---- distillation (loss) ----
distillation_loss_mode=k1
distillation_topk=64
use_policy_gradient=True
loss_max_clamp=10.0
log_prob_min_clamp=-10.0

# ---- trainer ----
project_name=verl_distill_mopd_gsm8k_geo3k
experiment_name=qwen3_vl_8b_from_qwen3_32b_and_qwen3_vl_32b_mopd_vllm_fsdp
total_epochs=15
save_freq=200
test_freq=5

# Multi-teacher: one teacher per dataset, routed by sample's `data_source` value.
# Use `+distillation.teacher_models.<name>.*` to add named teachers; the default `teacher_model`
# entry is silently popped when other teacher entries are added.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.image_key=images \
    actor_rollout_ref.model.path="$student_model" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$rollout_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_gpu_mem_util \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_model_len=$max_num_tokens \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    distillation.enabled=True \
    distillation.n_gpus_per_node=$teacher_world_size \
    distillation.nnodes=$teacher_nnodes \
    distillation.teacher_key=data_source \
    +distillation.teacher_models.gsm8k.key="openai/gsm8k" \
    +distillation.teacher_models.gsm8k.model_path="$gsm8k_teacher_model" \
    +distillation.teacher_models.gsm8k.num_replicas=$teacher_num_replicas_gsm8k \
    +distillation.teacher_models.gsm8k.inference.name=vllm \
    +distillation.teacher_models.gsm8k.inference.tensor_model_parallel_size=$teacher_tp \
    +distillation.teacher_models.gsm8k.inference.gpu_memory_utilization=$teacher_gpu_mem_util \
    +distillation.teacher_models.gsm8k.inference.max_model_len=$max_num_tokens \
    +distillation.teacher_models.geo3k.key="hiyouga/geometry3k" \
    +distillation.teacher_models.geo3k.model_path="$geo3k_teacher_model" \
    +distillation.teacher_models.geo3k.num_replicas=$teacher_num_replicas_geo3k \
    +distillation.teacher_models.geo3k.inference.name=vllm \
    +distillation.teacher_models.geo3k.inference.tensor_model_parallel_size=$teacher_tp \
    +distillation.teacher_models.geo3k.inference.gpu_memory_utilization=$teacher_gpu_mem_util \
    +distillation.teacher_models.geo3k.inference.max_model_len=$max_num_tokens \
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
