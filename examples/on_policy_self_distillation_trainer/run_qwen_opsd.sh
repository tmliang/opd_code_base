#!/usr/bin/env bash
# OPSD — faithful reproduction of the upstream OPSD recipe (run_opsd_4b.sh).
#   teacher  = base policy with LoRA disabled (== fixed reference)
#   teacher prompt = original problem + reference solution + transition prompt
#   loss     = forward KL = KL(teacher || student), top-K=128, per-token clamp 0.05
#   sampling = T=1.1 top_p=0.95 top_k=20, completion_length=1024
#   optim    = lr=5e-6, grad_clip=0.1
set -xeuo pipefail

# ---- model / data ----
model_name=/data2/haichao/opd_base/Qwen/Qwen3.5-2B
data_dir=data/openthoughts_math_30k_opsd
train_data=[${data_dir}/train.parquet]
val_data=[${data_dir}/test.parquet]

# ---- distributed ----
nnodes=1
n_gpus_per_node=8

# ---- batch / sequence ----
train_batch_size=32
ppo_mini_batch_size=32
max_prompt_length=4096
max_response_length=1024
ppo_max_token_len_per_gpu=24576

# ---- optim ----
actor_lr=5e-6
grad_clip=0.1

# ---- rollout (T=1.1 top_p=0.95 top_k=20) ----
rollout_n=1
rollout_tp=1
rollout_gpu_mem_util=0.5
rollout_temperature=1.1
rollout_top_p=0.95
rollout_top_k=20

# ---- distillation (loss) ----
# alpha=0 + tail=renorm + topk=128 + max_clamp=0.05 ≡ upstream JSD-clip(0.05) on K=128 renormalised
sdpo_alpha=0.0
topk=128
loss_max_clamp=0.05
distillation_loss_coef=1.0

# ---- self-distill (teacher) ----
teacher_dataloader=opsd
teacher_update=ref   # fixed reference

# ---- trainer ----
project_name=opsd_examples
experiment_name=opsd_qwen35_2b_faithful
total_epochs=30
save_freq=200
test_freq=5

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
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
    actor_rollout_ref.actor.grad_clip=$grad_clip \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$rollout_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_gpu_mem_util \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.temperature=$rollout_temperature \
    actor_rollout_ref.rollout.top_p=$rollout_top_p \
    actor_rollout_ref.rollout.top_k=$rollout_top_k \
    actor_rollout_ref.rollout.response_length=$max_response_length \
    distillation.enabled=True \
    distillation.mode=self \
    distillation.self_distill.dataloader="$teacher_dataloader" \
    distillation.self_distill.teacher_update=$teacher_update \
    distillation.distillation_loss.kl_family=sdpo \
    distillation.distillation_loss.sdpo.alpha=$sdpo_alpha \
    distillation.distillation_loss.sdpo.mode=topk \
    distillation.distillation_loss.topk=$topk \
    distillation.distillation_loss.sdpo.tail=renorm \
    distillation.distillation_loss.loss_max_clamp=$loss_max_clamp \
    distillation.distillation_loss.use_policy_gradient=False \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.distillation_loss_coef=$distillation_loss_coef \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.total_epochs=$total_epochs \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.logger='["console","swanlab"]' \
    "$@"
