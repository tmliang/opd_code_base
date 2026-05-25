#!/usr/bin/env bash
# Faithful SDPO recipe (https://github.com/siyan-zhao/SDPO) on the verl OPSD
# self-distillation framework.
#   teacher prompt = student prompt + successful same-uid sibling response via
#                    SDPO reprompt template
#   loss           = SDPO alpha-KL on teacher top-K (add_tail=True), IS-clipped@2.0
#   teacher        = EMA-tracking reference, decay = 1 - 0.05 = 0.95
set -xeuo pipefail

# ---- model / data ----
model_name=Qwen/Qwen3.5-2B
data_dir=$HOME/data/gsm8k
train_data=[${data_dir}/train.parquet]
val_data=[${data_dir}/test.parquet]

# ---- distributed ----
nnodes=1
n_gpus_per_node=8

# ---- batch / sequence ----
train_batch_size=32
ppo_mini_batch_size=32
max_prompt_length=1024
max_response_length=2048
ppo_max_token_len_per_gpu=24576

# ---- optim ----
actor_lr=1e-5
lr_warmup_steps=10

# ---- rollout ----
rollout_n=8
val_rollout_n=16
rollout_tp=2
rollout_gpu_mem_util=0.5

# ---- distillation (loss) — actor.yaml defaults ----
sdpo_alpha=0.5                       # 0=fwd KL, 0.5=JSD, 1=rev KL
topk=100                                   # distillation_topk
sdpo_ratio_clip=2.0             # ratio_clip

# ---- self-distill (teacher) ----
teacher_dataloader=sdpo
teacher_update=ema
ema_rate=0.05                          # teacher_update_rate
sdpo_success_threshold=0.5
sdpo_dont_reprompt_on_self_success=True
sdpo_strip_thinking=True    # remove_thinking_from_demonstration

# ---- rollout correction ----
rollout_is=token
rollout_is_threshold=2.0

# ---- trainer ----
project_name=opsd_examples
experiment_name=sdpo_qwen35_2b
total_epochs=15
save_freq=200
test_freq=5

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
    data.truncation=error \
    actor_rollout_ref.model.path="$model_name" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$lr_warmup_steps \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$rollout_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_gpu_mem_util \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.val_kwargs.n=$val_rollout_n \
    distillation.enabled=True \
    distillation.mode=self \
    distillation.self_distill.dataloader="$teacher_dataloader" \
    +distillation.self_distill.dataloader_kwargs.success_reward_threshold=$sdpo_success_threshold \
    +distillation.self_distill.dataloader_kwargs.dont_reprompt_on_self_success=$sdpo_dont_reprompt_on_self_success \
    +distillation.self_distill.dataloader_kwargs.remove_thinking_from_demonstration=$sdpo_strip_thinking \
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
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.total_epochs=$total_epochs \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    "$@"
