#!/usr/bin/env bash
# Vision-OPD recipe — faithful reproduction of yuanqianhao/Vision-OPD.
#   model    = Qwen3.5-VL (or any HF VL chat model)
#   teacher  = student with image pixels swapped to bbox-highlighted variant
#   loss     = SDPO alpha-KL on teacher top-K, alpha=0.5 (Jensen-Shannon)
#   teacher  = EMA-tracking reference (Vision-OPD key ingredient)
#   reward   = none for gradient (use_task_rewards=False); MCQ acc for logging only
set -xeuo pipefail

# ---- model / data ----
model_name=Qwen/Qwen3.5-2B                 # NOTE: needs a VL chat model
data_dir=data/vision_opd_local
train_data=[${data_dir}/Vision-OPD-6K.parquet]
val_data=[${data_dir}/vstar_bench.parquet]
image_key=images

# ---- distributed ----
nnodes=1
n_gpus_per_node=8

# ---- batch / sequence ----
train_batch_size=32
ppo_mini_batch_size=32
max_prompt_length=2048
max_response_length=4096
ppo_max_token_len_per_gpu=9216
max_num_batched_tokens=9216
max_model_len=9216
mm_min_pixels=4096
mm_max_pixels=131072

# ---- optim ----
actor_lr=2e-6
lr_warmup_steps=10
clip_ratio_low=0.2
clip_ratio_high=0.3

# ---- rollout ----
rollout_n=8
rollout_tp=1
rollout_gpu_mem_util=0.45
rollout_agent_workers=2

# ---- distillation (loss) ----
sdpo_alpha=0.5
topk=100
sdpo_ratio_clip=2.0
distillation_loss_coef=1.0

# ---- self-distill (teacher) ----
teacher_dataloader=vision_opd
teacher_update=ema
ema_rate=0.05

# ---- rollout correction ----
rollout_is=token
rollout_is_threshold=2.0

# ---- trainer ----
project_name=Vision-OPD
experiment_name=Vision-OPD-$(basename "$model_name")
total_epochs=1
save_freq=20
test_freq=20

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=$rollout_is \
    algorithm.rollout_correction.rollout_is_threshold=$rollout_is_threshold \
    data.train_files="$train_data" \
    data.val_files="$val_data" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    data.truncation=right \
    +data.mm_processor_kwargs.max_pixels=$mm_max_pixels \
    +data.mm_processor_kwargs.min_pixels=$mm_min_pixels \
    data.shuffle=True \
    data.trust_remote_code=True \
    data.return_multi_modal_inputs=True \
    data.image_key=$image_key \
    data.dataloader_num_workers=0 \
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
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$rollout_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_gpu_mem_util \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.max_model_len=$max_model_len \
    actor_rollout_ref.rollout.response_length=$max_response_length \
    actor_rollout_ref.rollout.agent.num_workers=$rollout_agent_workers \
    distillation.enabled=True \
    distillation.mode=self \
    distillation.self_distill.dataloader="$teacher_dataloader" \
    distillation.self_distill.teacher_update=$teacher_update \
    distillation.self_distill.teacher_update_rate=$ema_rate \
    distillation.distillation_loss.kl_family=sdpo \
    distillation.distillation_loss.sdpo.alpha=$sdpo_alpha \
    distillation.distillation_loss.sdpo.mode=topk \
    distillation.distillation_loss.sdpo.tail=add \
    distillation.distillation_loss.sdpo.ratio_clip=$sdpo_ratio_clip \
    distillation.distillation_loss.topk=$topk \
    distillation.distillation_loss.use_policy_gradient=False \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.distillation_loss_coef=$distillation_loss_coef \
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
