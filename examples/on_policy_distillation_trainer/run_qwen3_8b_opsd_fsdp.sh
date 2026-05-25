#!/usr/bin/env bash
# OPSD (On-Policy Self-Distillation) | text | vLLM rollout | FSDP training
#
# Differences vs. standard OPD:
#   * distillation.mode=self           -> teacher colocated with actor (no teacher pool)
#   * distillation.self_distill.dataloader -> registered short name (opsd/sdpo/vision_opd) or pkg.module:Class FQN
#   * No teacher_models / TEACHER_WORLD_SIZE / TEACHER_TP needed
#
# Before launching, implement a teacher dataloader subclassing
#   verl.workers.self_distillation.OfflineTeacherDataloader   OR
#   verl.workers.self_distillation.OnlineTeacherDataloader
# (see docs/algo/opsd.md).
set -xeuo pipefail

# ---- model / data ----
student_model=Qwen/Qwen3-8B

gsm8k_train=$HOME/data/gsm8k/train.parquet
gsm8k_test=$HOME/data/gsm8k/test.parquet
math_train=$HOME/data/math/train.parquet
math_test=$HOME/data/math/test.parquet
train_files=[$gsm8k_train,$math_train]
val_files=[$gsm8k_test,$math_test]

# ---- distributed ----
nnodes=1
n_gpus_per_node=8

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
rollout_gpu_mem_util=0.5
rollout_n=1                                  # >1 enables sibling-rollout recipes

# ---- distillation (loss) ----
# Estimator-mode losses: k1/k3/abs/mse/low_var_kl/kl
distillation_loss_mode=k3
use_policy_gradient=False
distillation_loss_coef=1.0
loss_max_clamp=10.0
log_prob_min_clamp=-10.0

# ---- self-distill (teacher) ----
# Either a registered short name (e.g. opsd / sdpo / vision_opd; see
# verl/workers/self_distillation/dataloaders/) OR a pkg.module:Class FQN of
# your own subclass of OfflineTeacherDataloader / OnlineTeacherDataloader.
teacher_dataloader_target="my_pkg.opsd:MyTeacherDataloader"
# teacher_update options:
#   ref          -> frozen reference teacher
#   ema          -> ref <- ema_decay*ref + (1-ema_decay)*actor every step (default)
#   progressive  -> ref <- actor every teacher_update_interval steps
#   trust_region -> ref <- actor when KL(student||teacher) <= trust_region_threshold
teacher_update=ema

# ---- trainer ----
project_name=verl_opsd_gsm8k_math
experiment_name=qwen3_8b_opsd_fsdp
total_epochs=15
save_freq=200
test_freq=5

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
    data.shuffle=False \
    actor_rollout_ref.model.path="$student_model" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$rollout_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_gpu_mem_util \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.max_model_len=$max_num_tokens \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    distillation.enabled=True \
    distillation.mode=self \
    distillation.self_distill.teacher_update=$teacher_update \
    distillation.self_distill.truncation=right \
    distillation.self_distill.dataloader=$teacher_dataloader_target \
    distillation.distillation_loss.loss_mode=$distillation_loss_mode \
    distillation.distillation_loss.use_task_rewards=True \
    distillation.distillation_loss.use_policy_gradient=$use_policy_gradient \
    distillation.distillation_loss.distillation_loss_coef=$distillation_loss_coef \
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
