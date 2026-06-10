#!/usr/bin/env bash
# Video-OPD recipe — time-reference-centred frame resampling on the OPSD framework.
#   model    = any HF video-capable VL chat model (Qwen2.5/3.5-VL etc.)
#   student  = standard pipeline, uniform frame sampling
#   teacher  = same text prompt, video re-decoded with frames concentrated on
#              extra_info.time_reference = [start_sec, end_sec] (answer span)
#   loss     = SDPO alpha-KL on teacher top-K, alpha=0.5 (Jensen-Shannon)
#   teacher  = EMA-tracking reference
#
# Data requirements (per row):
#   * prompt messages with one {"type": "video", "video": "<path or file://...>"} segment
#   * extra_info.time_reference: [start_sec, end_sec] (or {"start":, "end":})
#   * optional extra_info.<video_path_field> if the path is not in the messages
set -xeuo pipefail

cleanup_ray_on_exit=True

cleanup() {
    local exit_code=$?
    if [[ "${cleanup_ray_on_exit}" == "True" ]]; then
        ray stop --force || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT

# ---- model / data ----
model_name=Qwen/Qwen3.5-VL-4B
data_dir=data/video_opd_local
train_data=[${data_dir}/train.parquet]
val_data=[${data_dir}/val.parquet]

# ---- distributed ----
nnodes=1
n_gpus_per_node=8

# ---- batch / sequence ----
train_batch_size=16
ppo_mini_batch_size=8
max_prompt_length=8192
max_response_length=1024
ppo_max_token_len_per_gpu=10240
max_num_batched_tokens=10240
max_model_len=10240

# ---- optim ----
actor_lr=1e-6
lr_warmup_steps=10

# ---- rollout ----
rollout_n=4
rollout_temperature=1.0
rollout_top_p=0.95
rollout_tp=1
rollout_gpu_mem_util=0.45

# ---- distillation (loss) ----
sdpo_alpha=0.5
topk=100
sdpo_ratio_clip=2.0

# ---- self-distill (teacher) ----
teacher_dataloader=video_opd
teacher_update=ema
ema_rate=0.05
# video_opd dataloader knobs:
focus_ratio=0.6          # fraction of frames inside the (widened) reference interval
context_margin_sec=2.0   # widen [start, end] by this many seconds per side
# num_frames defaults to the student's frame count; uncomment to override:
# num_frames=16

# ---- trainer ----
project_name=Video-OPD
run_timestamp=$(date +%Y%m%d_%H%M%S)
experiment_name=Video-OPD-$(basename "$model_name")-${run_timestamp}
total_epochs=1
save_freq=20
test_freq=20

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_data" \
    data.val_files="$val_data" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation=right \
    data.return_multi_modal_inputs=True \
    actor_rollout_ref.model.path="$model_name" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$lr_warmup_steps \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$rollout_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_gpu_mem_util \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.temperature=$rollout_temperature \
    actor_rollout_ref.rollout.top_p=$rollout_top_p \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.max_model_len=$max_model_len \
    actor_rollout_ref.rollout.response_length=$max_response_length \
    distillation.enabled=True \
    distillation.mode=self \
    distillation.self_distill.dataloader="$teacher_dataloader" \
    distillation.self_distill.teacher_update=$teacher_update \
    distillation.self_distill.teacher_update_rate=$ema_rate \
    +distillation.self_distill.dataloader_kwargs.focus_ratio=$focus_ratio \
    +distillation.self_distill.dataloader_kwargs.context_margin_sec=$context_margin_sec \
    distillation.distillation_loss.kl_family=sdpo \
    distillation.distillation_loss.sdpo.alpha=$sdpo_alpha \
    distillation.distillation_loss.sdpo.mode=topk \
    distillation.distillation_loss.sdpo.tail=add \
    distillation.distillation_loss.sdpo.ratio_clip=$sdpo_ratio_clip \
    distillation.distillation_loss.topk=$topk \
    distillation.distillation_loss.use_policy_gradient=False \
    distillation.distillation_loss.use_task_rewards=False \
    reward_model.enable=False \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.total_epochs=$total_epochs \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.val_before_train=False \
    trainer.logger='["console","swanlab"]' \
    "$@"
