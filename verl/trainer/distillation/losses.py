# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    if response_mask.is_nested:
        distillation_losses_response = distillation_losses[response_mask.bool().to_padded_tensor(False)]
    else:
        distillation_losses_response = distillation_losses[response_mask.bool()]
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    loss_mode = distillation_config.distillation_loss.loss_mode
    is_reverse_kl_topk = loss_mode == "reverse_kl_topk"
    match config.strategy:
        # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
        case "fsdp" | "veomni":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            if loss_mode == "sdpo_alpha_kl_topk":
                distillation_loss_fn = fsdp_losses.compute_sdpo_alpha_kl_topk
            elif is_reverse_kl_topk:
                distillation_loss_fn = fsdp_losses.compute_reverse_kl_topk
            else:
                distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            import verl.trainer.distillation.megatron.losses as megatron_losses

            if loss_mode == "sdpo_alpha_kl_topk":
                raise NotImplementedError(
                    "sdpo_alpha_kl_topk is not yet implemented for the Megatron backend; "
                    "use kl_family=verl or strategy=fsdp."
                )
            if is_reverse_kl_topk:
                raise NotImplementedError(
                    "reverse_kl_topk is currently FSDP-only. Use strategy=fsdp."
                )
            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    extra_kwargs = {}
    if is_reverse_kl_topk and distillation_config.distillation_loss.use_tail_sampling:
        # Teacher logprob at the student-sampled token. Accept either canonical
        # key; OPSD's topk teacher pipeline emits `teacher_sampled_logprob`.
        td_keys = data.keys(include_nested=True) if hasattr(data, "keys") else data.keys()
        if "sampled_teacher_logprob" in td_keys:
            extra_kwargs["sampled_teacher_logprob"] = data["sampled_teacher_logprob"]
        elif "teacher_sampled_logprob" in td_keys:
            extra_kwargs["sampled_teacher_logprob"] = data["teacher_sampled_logprob"]
        else:
            raise KeyError(
                "reverse_kl_topk with use_tail_sampling=True requires "
                "data['sampled_teacher_logprob'] (or 'teacher_sampled_logprob'): teacher's logprob "
                "at the student-sampled token, shape (B, response_len). For OPSD this is produced "
                "automatically by the topk teacher path; for OPD configure the teacher rollout to "
                "surface per-sampled-token logprobs and merge them into the batch under that key."
            )
        extra_kwargs["sampled_token_ids"] = data["responses"]

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
        **extra_kwargs,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        # Skip scalar (0-dim) auxiliaries such as not_in_topk_ratio.
        if v.dim() == 0:
            continue
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
    if not distillation_loss_config.use_task_rewards:
        policy_loss = 0.0

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    return policy_loss, policy_metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    # OPD trick (revisiting_opd §4.3): mask first occurrence of given special
    # tokens in each response. Mitigates spurious KL spikes when the student
    # and teacher tokenize chat templates / reasoning markers slightly
    # differently. Default off.
    if loss_config.opd_mask_special_tokens and loss_config.opd_mask_token_ids:
        from verl.utils.distillation.special_token_mask import build_first_occurrence_mask

        mask = build_first_occurrence_mask(
            responses=data["responses"],
            response_mask=response_mask,
            token_ids=list(loss_config.opd_mask_token_ids),
        )
        # Replace response_mask with the masked version for both metrics and aggregation.
        response_mask = mask
        # Also stash for downstream consumers (e.g. policy-loss path).
        data["response_mask"] = mask

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    # SDPO-style per-token importance ratio clip:
    # mul each token loss by min(exp(student_lp - old_lp), ratio_clip).
    if loss_config.kl_family == "sdpo" and loss_config.sdpo.ratio_clip is not None:
        is_clip = float(loss_config.sdpo.ratio_clip)
        student_lp = no_padding_2_padding(model_output["log_probs"], data)
        old_lp = data["old_log_probs"]
        if old_lp.is_nested:
            old_lp = old_lp.to_padded_tensor(0.0)
        approx_kl = (student_lp - old_lp).detach().clamp(min=-20.0, max=20.0)
        ratio = torch.exp(approx_kl).clamp(max=is_clip)
        distillation_losses = distillation_losses * ratio
        distillation_metrics["distillation/sdpo_ratio_clip_mean"] = Metric(
            AggregationType.MEAN, ratio.mean()
        )

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        rollout_is_weights = data.get("rollout_is_weights", None)
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["reverse_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_reverse_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reverse KL divergence over teacher top-K.

    Ported from `revisiting_opd` (compute_memory_efficient_kl with
    kl_type='full_reverse'). Combines:

    * **Head term** ``L1`` -- KL(pi_student || pi_teacher) restricted to the
      teacher's top-K vocabulary positions, optionally renormalized to
      sum-to-1 over K.
    * **Tail term** ``L2`` -- correction over the student-sampled token when
      it falls outside the teacher's top-K. Activated by
      ``distillation_loss.use_tail_sampling=True``.

    Optional importance-sampling reweighting (``use_kl_iw``) is applied to
    ``L2`` using ``exp(log_pi - log_pi_old)`` clipped to
    ``[kl_iw_clip_lower, kl_iw_clip_upper]``.
    """
    loss_config: DistillationLossConfig = distillation_config.distillation_loss

    L1 = no_padding_2_padding(model_output["distillation_losses_head"], data)
    L2 = no_padding_2_padding(model_output["distillation_losses_tail"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)

    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert L1.shape == L2.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    # Apply optional importance-sampling reweighting to the tail term.
    if loss_config.use_tail_sampling and loss_config.use_kl_iw:
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = old_log_prob.to_padded_tensor(0.0)
        log_iw = (log_prob - old_log_prob).detach().clamp(min=-20.0, max=20.0)
        iw = log_iw.exp()
        if loss_config.kl_iw_clip_lower is not None or loss_config.kl_iw_clip_upper is not None:
            iw = iw.clamp(min=loss_config.kl_iw_clip_lower, max=loss_config.kl_iw_clip_upper)
        L2 = L2 * iw

    distillation_losses = L1 + L2

    # Reverse KL over a top-K-restricted distribution can dip slightly below
    # zero when ``norm_to_one_for_kl=False`` (since the partial probabilities
    # do not sum to 1). Clamp to >=0 for stability of policy-gradient mode.
    distillation_losses = distillation_losses.clamp_min(0.0)

    student_mass_v = student_mass[response_mask_bool]
    teacher_mass_v = teacher_mass[response_mask_bool]
    metrics: dict[str, Any] = {
        "distillation/student_mass": student_mass_v.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass_v.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass_v.max()),
        "distillation/teacher_mass": teacher_mass_v.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass_v.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass_v.max()),
        "distillation/head_loss": Metric(
            AggregationType.MEAN, L1[response_mask_bool].mean() if response_mask_bool.any() else L1.new_zeros(())
        ),
    }
    if loss_config.use_tail_sampling:
        metrics["distillation/tail_loss"] = Metric(
            AggregationType.MEAN, L2[response_mask_bool].mean() if response_mask_bool.any() else L2.new_zeros(())
        )
        # Optional scalar logged from the kernel.
        if "not_in_topk_ratio" in model_output:
            ratio = model_output["not_in_topk_ratio"]
            if isinstance(ratio, torch.Tensor):
                ratio = ratio.detach().float().mean()
            metrics["distillation/not_in_topk_ratio"] = Metric(AggregationType.MEAN, ratio)

    return distillation_losses, metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss.
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }
    return distillation_losses, metrics


@register_distillation_loss(DistillationLossSettings(names=["sdpo_alpha_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_distillation_loss_sdpo_alpha_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """SDPO alpha-KL on top-k log-probs. The heavy lifting happens inside the
    logits processor (see ``compute_topk_loss``); this function just unpacks
    the precomputed per-token loss and exposes mass metrics.
    """
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()

    student_mass_sel = student_mass[response_mask_bool]
    teacher_mass_sel = teacher_mass[response_mask_bool]
    sdpo_cfg = distillation_config.distillation_loss.sdpo
    metrics = {
        "distillation/sdpo_alpha": float(sdpo_cfg.alpha),
        "distillation/student_mass": student_mass_sel.mean().item(),
        "distillation/teacher_mass": teacher_mass_sel.mean().item(),
    }
    # Numerical floors: small negative values can leak from top-k truncation
    # without add_tail/renorm, or from fp16 noise.
    distillation_losses = distillation_losses.clamp_min(0.0)
    return distillation_losses, metrics


@register_distillation_loss(
    DistillationLossSettings(names=["sdpo_alpha_kl_sampled"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_sdpo_alpha_kl_sampled(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """SDPO alpha-KL on sampled tokens. Only ``alpha == 1.0`` (reverse KL) is
    well-defined here — without distributions we can't form the JSD mixture.
    This kernel therefore reduces to verl's existing reverse-KL estimator
    (k1 / k3 family, picked via ``sdpo.sampled_estimator``-style override is
    not exposed; defaults to ``k3``).
    """
    sdpo_cfg = distillation_config.distillation_loss.sdpo
    if sdpo_cfg.alpha != 1.0:
        raise NotImplementedError(
            "sdpo_alpha_kl_sampled only supports alpha=1.0 (reverse KL). "
            "Set sdpo.mode=topk (or full, once wired) for alpha != 1.0."
        )
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty="k3"
    )
    metrics = {
        "distillation/sdpo_alpha": 1.0,
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }
    return distillation_losses, metrics
