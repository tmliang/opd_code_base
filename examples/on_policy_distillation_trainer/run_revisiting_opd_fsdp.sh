#!/usr/bin/env bash
# Revisiting-OPD recipe ported to the new verl (3 paper tricks on top of OPD).
#   Reference: https://github.com/hhh675597/revisiting_opd
#   Backend  : FSDP actor + ref, vLLM rollout + teacher inference
#   For multi-teacher add entries under `distillation.teacher_models.*`.
set -xeuo pipefail

# ---- model / data ----
student_model=Qwen/Qwen2.5-7B-Instruct
teacher_model=/path/to/OpenThinker3-7B
train_data=[/path/to/math_opd/train.parquet]
val_data=[/path/to/math_opd/test.parquet]

# ---- distributed ----
nnodes=1
n_gpus_per_node=8
teacher_world_size=4

# ---- batch / sequence ----
train_batch_size=16
ppo_mini_batch_size=64
max_prompt_length=2048
max_response_length=16384
ppo_max_token_len_per_gpu=24576
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

# ---- optim ----
actor_lr=2e-6

# ---- rollout (trick 5: top-p sampling) ----
rollout_tp=1
rollout_gpu_mem_util=0.5
rollout_top_p=0.9

# ---- teacher ----
teacher_tp=2
teacher_gpu_mem_util=0.5

# ---- distillation (loss) — defaults preserve vanilla forward_kl_topk ----
distillation_loss_mode=reverse_kl_topk
distillation_topk=32
use_policy_gradient=False     # paper uses pure supervised KL
loss_max_clamp=10.0
log_prob_min_clamp=-10.0

# ---- revisiting_opd tricks (opt-in) ----
norm_to_one=True                      # trick 1: renormalize over K
clip_log_ratio=False               # trick 1: clip (log p - log q)
use_tail_sampling=False         # trick 2: head + tail
use_kl_iw=False                         # trick 3: IW on tail
kl_iw_clip_lower=0.0
kl_iw_clip_upper=10.0
opd_mask_special=True            # trick 4: mask first <think> etc.

# ---- trainer ----
project_name=revisiting_opd
experiment_name=math_qwen2.5_7b_it_reverse_kl_topk
total_epochs=1
save_freq=-1
test_freq=40

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_data" \
    data.val_files="$val_data" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='middle' \
    data.shuffle=False \
    actor_rollout_ref.model.path="$student_model" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$rollout_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_gpu_mem_util \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_model_len=$max_num_tokens \
    actor_rollout_ref.rollout.top_p=$rollout_top_p \
    actor_rollout_ref.rollout.val_kwargs.top_p=$rollout_top_p \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
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
    distillation.distillation_loss.norm_to_one_for_kl=$norm_to_one \
    distillation.distillation_loss.clip_log_ratio=$clip_log_ratio \
    distillation.distillation_loss.use_tail_sampling=$use_tail_sampling \
    distillation.distillation_loss.use_kl_iw=$use_kl_iw \
    distillation.distillation_loss.kl_iw_clip_lower=$kl_iw_clip_lower \
    distillation.distillation_loss.kl_iw_clip_upper=$kl_iw_clip_upper \
    distillation.distillation_loss.opd_mask_special_tokens=$opd_mask_special \
    distillation.distillation_loss.opd_mask_first_tokens='["<","think","<|im_end|>"]' \
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
