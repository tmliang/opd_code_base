# PR — Port `revisiting_opd` tricks into the new verl OPD pipeline

> Source repo (old, forked from pre-multitask verl):
> [`/mmu_mllm_hdd_3/liangtianming/revisiting_opd`](../../revisiting_opd) ·
> paper: *Revisiting On-Policy Distillation: Empirical Failure Modes and
> Simple Fixes*.
>
> Target repo (current verl with full OPD scaffolding):
> [`/mmu_mllm_hdd_3/liangtianming/verl`](..) — `verl/trainer/distillation/`,
> `verl/workers/config/distillation.py`,
> `examples/on_policy_distillation_trainer/`.

## TL;DR

The new verl already ships an OPD pipeline (forward-KL top-K loss, multi-teacher
serving via vLLM/SGLang, async schedulers, FSDP+Megatron kernels). What it was
**missing** are the specific *training tricks* introduced by `revisiting_opd`.
This PR ports them, with every trick gated by a config flag (defaults preserve
current behaviour, no breaking changes).

| # | Trick (paper §) | Status before | This PR |
|---|---|---|---|
| 1 | Reverse-KL over teacher top-K (`compute_memory_efficient_kl`) | ❌ — only `forward_kl_topk` existed | ✅ new `reverse_kl_topk` registered loss + FSDP kernel |
| 2 | Tail-sampling correction (head + tail with union-of-K+1) | ❌ | ✅ `use_tail_sampling` flag |
| 3 | Importance-weight reweighting on tail term | ❌ | ✅ `use_kl_iw` + `kl_iw_clip_{lower,upper}` |
| 4 | Special-token first-occurrence masking (`<think>`, `</think>`, `<\|im_end\|>` …) | ❌ | ✅ `opd_mask_special_tokens` + auto-encode at trainer init |
| 5 | Top-p rollouts | ✅ already in `RolloutConfig.top_p` | — (documented in example) |
| 6 | Multi-task ref worker (`multitask_ref_worker.py`) | ✅ subsumed by `MultiTeacherModelManager` (new verl) | — (no port needed) |
| 7 | "Placeholder" advantage estimator | ✅ subsumed by `use_task_rewards=False` + `use_policy_gradient=False` | — (no port needed) |
| 8 | Distribution visualization utility | ❌ | ✅ ported verbatim as opt-in util |

## Files touched

| File | Change | Why |
|---|---|---|
| [verl/workers/config/distillation.py](../../verl/workers/config/distillation.py) | + 8 new fields on `DistillationLossConfig` (`norm_to_one_for_kl`, `clip_log_ratio`, `use_tail_sampling`, `use_kl_iw`, `kl_iw_clip_lower`, `kl_iw_clip_upper`, `opd_mask_special_tokens`, `opd_mask_first_tokens`, `opd_mask_token_ids`). Marked `opd_mask_token_ids` mutable. | Opt-in toggles for every trick; defaults preserve previous behaviour. |
| [verl/trainer/distillation/fsdp/losses.py](../../verl/trainer/distillation/fsdp/losses.py) | + `compute_reverse_kl_topk` kernel | Head-only reverse KL on teacher top-K, with optional tail term using the union-of-(K+1) reformulation. |
| [verl/trainer/distillation/losses.py](../../verl/trainer/distillation/losses.py) | dispatch `reverse_kl_topk` in `compute_topk_loss`; register `reverse_kl_topk` distillation loss; apply special-token mask inside `distillation_loss` | Wire kernel → registry, combine head+IW·tail, plug in masking. |
| [verl/utils/distillation/__init__.py](../../verl/utils/distillation/__init__.py) | + new package | Re-exports the helpers. |
| [verl/utils/distillation/special_token_mask.py](../../verl/utils/distillation/special_token_mask.py) | + `encode_special_token_ids`, `build_first_occurrence_mask` | Tokeniser → ids; per-token first-occurrence mask matching old semantics. |
| [verl/utils/distillation/visualize_distribution.py](../../verl/utils/distillation/visualize_distribution.py) | + ported util | Interactive HTML diff of teacher vs student token probs. Not wired into trainer. |
| [verl/trainer/main_ppo_sync.py](../../verl/trainer/main_ppo_sync.py) | auto-encode `opd_mask_first_tokens` → `opd_mask_token_ids` using trainer tokenizer | UX: user can keep using string lists; ids are filled automatically once. |
| [examples/on_policy_distillation_trainer/run_revisiting_opd_fsdp.sh](../../examples/on_policy_distillation_trainer/run_revisiting_opd_fsdp.sh) | + new example | Reference recipe that exposes every new flag as an env var. |
| [docs/advance/revisiting_opd_pr.md](./revisiting_opd_pr.md) | + this file | Migration notes. |

No existing files were renamed, removed, or had public APIs broken.

## Design notes

### 1. `reverse_kl_topk` math

Faithfully mirrors `revisiting_opd/verl/trainer/ppo/core_algos.py::compute_memory_efficient_kl`
(kl_type=`full_reverse`). Two branches, controlled by
`distillation.distillation_loss.norm_to_one_for_kl`:

* **`norm_to_one_for_kl=True`** (default & paper-recommended): renormalise the
  top-K slice so that it sums to 1 before computing the KL. This is the
  "memory-efficient full reverse KL" the paper calls "Teacher-TopK".
  $$L_1 = \sum_{v\in\text{topK}} \tilde\pi_s(v)\,\big(\log\tilde\pi_s(v) - \log\tilde\pi_t(v)\big),$$
  with $\tilde\pi = \text{softmax}(\log\pi|_\text{topK})$.
* **`norm_to_one_for_kl=False`**: use the partial probabilities directly (do
  NOT renormalize). Equivalent to a sparse KL where the K probabilities do
  not sum to 1.

`clip_log_ratio=True` clips $\log\tilde\pi_s - \log\tilde\pi_t$ to $[-5, 5]$
inside the kernel for stability.

### 2. Tail sampling (head + tail)

When `use_tail_sampling=True`, the kernel additionally takes
`sampled_token_ids` (= `data["responses"]`) and `sampled_teacher_logprob`
(supplied by the caller in `data["sampled_teacher_logprob"]`, shape
`(B, response_len)`). It builds a K+1 union per token using log-softmax's
shift-invariance:

```text
union_student[:, :, :K] = student_topk_log_probs           # at teacher top-K
union_student[:, :,  K] = student_logprob_at_sampled if sampled∉topK else -inf
union_teacher[:, :, :K] = teacher_topk_log_probs           # given
union_teacher[:, :,  K] = teacher_logprob_at_sampled if sampled∉topK else -inf
norm_*  = log_softmax(union_*, dim=-1)
L1 = (exp(s_norm[..., :K]) * diff[..., :K]).sum(-1)        # head
L2 = exp(s_norm[..., K]) * diff[..., K]                    # tail (zero when in-topK)
```

The `-inf` slot makes the tail term automatically zero whenever the sampled
token is already inside teacher's top-K, so `L2` is only nonzero where the
correction is actually needed.

> **Data contract.** The teacher's logprob at the student-sampled token is
> *not* part of the existing `teacher_logprobs` / `teacher_ids` payload (those
> are the top-K, which may exclude the sampled token). Producing it requires
> the teacher rollout to also score the sampled tokens (e.g. vLLM
> `prompt_logprobs=1` over student-sampled positions, or scoring the response
> in the FSDP ref worker). Users enabling `use_tail_sampling=True` must
> populate `data["sampled_teacher_logprob"]`; otherwise the loss raises a
> clear `KeyError`. Implementing the teacher-side scoring is intentionally
> left out of this PR (it depends on the deployment topology — vLLM teacher
> vs. async server vs. FSDP ref worker). See "Follow-ups" below.

### 3. Importance weighting on tail

When `use_kl_iw=True`, the IW factor
$\rho_t = \exp(\log\pi_\theta(a_t) - \log\pi_{\theta_\text{old}}(a_t))$
is computed inside the registered loss (it needs `model_output["log_probs"]`
and `data["old_log_probs"]`, which are not visible from the FSDP kernel),
clipped to `[kl_iw_clip_lower, kl_iw_clip_upper]`, and multiplied into the
tail term:

```text
distillation_losses = L1 + ρ · L2
```

The head term is left unweighted, matching the old code.

### 4. Special-token first-occurrence masking

Implements the paper's §4.3 fix for tokenizer-drift artefacts. Inside
`distillation_loss()` we replace `data["response_mask"]` with the masked
version *before* aggregation (so both `agg_loss` and the policy-gradient
branch see the corrected mask). The mask helper preserves the original
semantics: **for each token id in the configured list, the first occurrence
per sample is masked**, not the first occurrence of *any* of them.

Auto-encoding (string → ids) is performed once, in
`RayPPOTrainerSync._init_workers()` right after `distillation_config` is
materialised, using the trainer's already-loaded tokenizer. Users can also
bypass auto-encoding by setting
`+distillation.distillation_loss.opd_mask_token_ids=[151644,…]` directly.

### 5. Visualisation utility

Ported verbatim from the old repo to
`verl/utils/distillation/visualize_distribution.py`. **Not** wired into the
trainer loop — it would require additional teacher-vs-student logprob plumbing
that's out of scope here. Call it from your own callback or a debug script:

```python
from verl.utils.distillation import visualize_teacher_student_batch
visualize_teacher_student_batch(batch, teacher_lp, student_lp, tokenizer,
                                global_step=step, output_dir="...", num_samples=2)
```

### 6. Tricks that did NOT need porting

* **Multi-task ref worker** (`multitask_ref_worker.py`). The new verl
  already routes per-key teachers via
  [`verl/experimental/teacher_loop/teacher_manager.py`](../../verl/experimental/teacher_loop/teacher_manager.py)
  (`MultiTeacherModelManager`). Configure with
  ```yaml
  distillation:
    teacher_models:
      math:
        key: math
        model_path: /path/to/math_teacher
        num_replicas: 1
      alfworld:
        key: alfworld
        model_path: /path/to/alfworld_teacher
        num_replicas: 1
  ```
  and tag each sample with its `key` in the dataset — same UX as the old
  `multitask.tasks[…].ref_model_path` block.

* **Placeholder advantage estimator** (`adv_estimator=placeholder` from the
  old repo). In the new verl, set
  `distillation.distillation_loss.use_task_rewards=False` and
  `distillation.distillation_loss.use_policy_gradient=False` to get pure
  supervised distillation (no PG, no advantages).

* **Top-p rollouts**. Already wired:
  `actor_rollout_ref.rollout.top_p=0.9` (+ matching `val_kwargs`).

## How to invoke

```bash
# 1. Plain Teacher-TopK reverse-KL (paper default, head only, masked).
DISTILLATION_LOSS_MODE=reverse_kl_topk \
DISTILLATION_TOPK=32 \
NORM_TO_ONE=True CLIP_LOG_RATIO=False \
USE_TAIL_SAMPLING=False USE_KL_IW=False \
OPD_MASK_SPECIAL=True ROLLOUT_TOP_P=0.9 \
USE_POLICY_GRADIENT=False \
bash examples/on_policy_distillation_trainer/run_revisiting_opd_fsdp.sh

# 2. Head + tail with IW (requires data["sampled_teacher_logprob"] -- see notes).
USE_TAIL_SAMPLING=True USE_KL_IW=True \
KL_IW_CLIP_LOWER=0 KL_IW_CLIP_UPPER=10 \
bash examples/on_policy_distillation_trainer/run_revisiting_opd_fsdp.sh
```

Every trick can be flipped independently via the env vars exposed by the
example script (or via plain Hydra overrides on the `distillation.distillation_loss.*`
namespace).

## Tests / validation

* `compute_reverse_kl_topk` kernel was sanity-checked against a manual
  top-K reverse-KL implementation for both `norm_to_one_for_kl ∈ {True, False}`:
  difference vs. reference is `< 1e-5` (machine precision). With
  `use_tail_sampling=True` the tail term is exactly `0` for positions where
  the sampled token lies inside teacher's top-K (verified empirically).
* `build_first_occurrence_mask` was tested with a 2-row tensor including
  both repeated and distinct target tokens; it masks only the first
  occurrence of each id per row, matching the old implementation.
* All edited files pass `pyright`-equivalent static checks (no errors
  reported by the IDE's diagnostics).

A full E2E test (FSDP + vLLM teacher + actual training) was not run in this
porting PR — see "Follow-ups".

## Follow-ups (out of scope for this PR)

1. **Teacher-side per-sample logprob** to make `use_tail_sampling=True`
   usable end-to-end. Two options:
   * Extend the vLLM teacher worker to also return `prompt_logprobs=1` over
     the student-sampled positions, then merge into the batch as
     `sampled_teacher_logprob`.
   * Or piggy-back on the existing `ref_policy_wg.compute_ref_log_prob`
     path (FSDP-based ref) and write its output to that key.
2. **Megatron kernel** for `reverse_kl_topk` (FSDP-only here per scope).
3. **Trainer-side visualisation hook** that periodically calls
   `visualize_teacher_student_batch` when
   `+trainer.visualize_distribution=true` (matches old recipe's flag).
4. **Reference-side top-K source** (`kl_topk_source=actor` from the old
   repo, where the student picks the top-K instead of the teacher).
   Currently `reverse_kl_topk` always uses teacher's top-K (`source=ref`),
   which matches the paper's recommended setting.

## Backward compatibility

* All new fields default to values that match prior behaviour
  (`opd_mask_special_tokens=False`, `use_tail_sampling=False`,
  `use_kl_iw=False`, etc.).
* The existing `forward_kl_topk` / `k1` / `k3` paths are untouched.
* `compute_topk_loss` now reads `loss_mode` from `distillation_config` to
  decide which kernel to dispatch; the previous default (`forward_kl_topk`)
  is taken unchanged for any non-`reverse_kl_topk` loss mode.
* The scalar `not_in_topk_ratio` output is filtered out of the per-token
  shape assertion in `compute_topk_loss` — no change for existing kernels
  whose outputs are all `(B, T)`.
