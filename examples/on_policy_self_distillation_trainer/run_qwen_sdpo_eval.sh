#!/usr/bin/env bash
# Evaluate one merged HF checkpoint on a chosen validation/test parquet.
#
# Usage:
#   bash examples/on_policy_self_distillation_trainer/run_qwen_sdpo_eval.sh \
#       checkpoints/sdpo_gsm8k/gt_opsd_sdpo_qwen35_2b/global_step_435/hf_merged \
#       data/gsm8k/test.parquet
#
# Equivalent environment overrides:
#   CKPT_PATH=... TEST_DATA=... bash examples/on_policy_self_distillation_trainer/run_qwen_sdpo_eval.sh
set -xeuo pipefail

# Clean up Ray/vLLM worker processes after eval or Ctrl-C so GPU memory is not
# held by orphaned workers. Disable if sharing the same node with another Ray job.
cleanup_ray_on_exit=True

cleanup() {
    local exit_code=$?
    if [[ "${cleanup_ray_on_exit}" == "True" ]]; then
        ray stop --force || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT

# ---- inputs ----
# model_path=${CKPT_PATH:-/data2/haichao/opd_hc/Qwen/Qwen3.5-2B}
model_path=${CKPT_PATH:-/data2/haichao/opd_hc/checkpoints/sdpo_gsm8k/gt_opsd_sdpo_qwen35_2b/global_step_435/hf_merged}

test_data=${TEST_DATA:-data/gsm8k/test.parquet}

if [[ $# -gt 0 && "$1" != *=* ]]; then
    model_path=$1
    shift
fi
if [[ $# -gt 0 && "$1" != *=* ]]; then
    test_data=$1
    shift
fi

# Eval should load a HF model path directly, not resume trainer state.
model_basename=$(basename "$model_path")
if [[ "$model_basename" == global_step_* ]]; then
    model_path="${model_path}/hf_merged"
elif [[ "$model_basename" == "actor" ]]; then
    model_path="$(dirname "$model_path")/hf_merged"
fi
if [[ -d "$model_path" && ! -f "$model_path/config.json" ]]; then
    echo "model_path is not a HF model directory: $model_path/config.json is missing" >&2
    exit 1
fi
if [[ ! -f "$test_data" ]]; then
    echo "test parquet does not exist: $test_data" >&2
    exit 1
fi

# ---- model / data ----
train_data=[$test_data]
val_data=[$test_data]

# ---- distributed ----
nnodes=1
n_gpus_per_node=8

# ---- batch / sequence ----
train_batch_size=256
val_batch_size=512
ppo_mini_batch_size=16
max_prompt_length=1024
max_response_length=8192
ppo_max_token_len_per_gpu=7168
max_model_len=9216
max_num_batched_tokens=7168
max_num_seqs=128

# ---- optim/model ----
actor_lr=1e-6
lr_warmup_steps=10
attn_implementation=flash_attention_2
activation_offload=False

# ---- rollout / validation sampling ----
rollout_n=1
rollout_do_sample=True
rollout_temperature=0.1
rollout_top_p=0.95
rollout_top_k=-1
val_rollout_n=1
val_do_sample=False
val_temperature=0.7
val_top_p=0.95
val_top_k=-1
repetition_penalty=${REPETITION_PENALTY:-1.1}
rollout_tp=1
rollout_gpu_mem_util=0.6
rollout_enforce_eager=False

# ---- trainer ----
project_name=sdpo_gsm8k
experiment_name=eval_gt_opsd_sdpo_qwen35_2b
validation_shuffle=False
validation_data_dir=outputs/${project_name}/${experiment_name}/train

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_data" \
    data.val_files="$val_data" \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.validation_shuffle=$validation_shuffle \
    data.truncation=error \
    actor_rollout_ref.model.path="$model_path" \
    +actor_rollout_ref.model.override_config.attn_implementation=$attn_implementation \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=$activation_offload \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$lr_warmup_steps \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.checkpoint.load_contents="['model']" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$rollout_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_gpu_mem_util \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.do_sample=$rollout_do_sample \
    actor_rollout_ref.rollout.temperature=$rollout_temperature \
    actor_rollout_ref.rollout.top_p=$rollout_top_p \
    actor_rollout_ref.rollout.top_k=$rollout_top_k \
    ++actor_rollout_ref.rollout.repetition_penalty=$repetition_penalty \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.max_model_len=$max_model_len \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.max_num_seqs=$max_num_seqs \
    actor_rollout_ref.rollout.enforce_eager=$rollout_enforce_eager \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.rollout.val_kwargs.n=$val_rollout_n \
    actor_rollout_ref.rollout.val_kwargs.do_sample=$val_do_sample \
    actor_rollout_ref.rollout.val_kwargs.temperature=$val_temperature \
    actor_rollout_ref.rollout.val_kwargs.top_p=$val_top_p \
    actor_rollout_ref.rollout.val_kwargs.top_k=$val_top_k \
    distillation.enabled=False \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.resume_mode=disable \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.validation_data_dir="$validation_data_dir" \
    trainer.logger='["console"]' \
    "$@"
