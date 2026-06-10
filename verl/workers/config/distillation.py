# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from verl.utils.config import omega_conf_to_dataclass

from .rollout import RolloutConfig

__all__ = [
    "DistillationLossConfig",
    "SDPODistillationLossConfig",
    "DistillationTeacherModelConfig",
    "DistillationConfig",
    "SelfDistillationConfig",
]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class SDPODistillationLossConfig(BaseConfig):
    """SDPO-style alpha-parameterized KL distillation loss.

    Ported from https://github.com/.../SDPO. Three classic KL families are
    expressed via a single interpolation coefficient ``alpha``:

    - ``alpha == 0.0``  → forward KL: ``KL(teacher || student)``
    - ``alpha == 0.5``  → symmetric Jensen-Shannon divergence
    - ``alpha == 1.0``  → reverse KL: ``KL(student || teacher)``
    - ``alpha in (0,1)`` (excl. 0.5) → generalized / skew JSD:
      ``(1-α)·KL(m||student) + α·KL(m||teacher)``  with
      ``m = (1-α)·p_s + α·p_t``.

    Consulted only when ``DistillationLossConfig.kl_family == "sdpo"``.

    alpha (float): KL interpolation coefficient in ``[0, 1]``.
    mode (str): ``"topk"`` (requires teacher top-k logprobs+ids in the batch,
        like the OPD external-teacher path) or ``"full"`` (full-vocab teacher
        logprobs; not yet wired). ``"sampled"`` is implicit when ``alpha=1``
        and reduces to the existing verl reverse-KL estimator path.
    tail (str): How to handle the residual probability mass outside the
        teacher top-K when ``mode == "topk"``. One of:

        - ``"add"`` (default): append a virtual K+1 bucket carrying
          ``log(1 - Σ p_i)`` to both distributions before computing the KL.
        - ``"renorm"``: re-normalize both top-K distributions to sum to 1.
        - ``"drop"``: use the raw top-K log-probs unchanged.
    ratio_clip (float, optional): PPO-style upper bound on the per-token
        importance ratio ``exp(student_logprob - old_logprob)`` used as a
        multiplier on the per-token distillation loss. ``None`` disables it.
    """

    alpha: float = 1.0
    mode: str = "topk"
    tail: str = "add"
    ratio_clip: Optional[float] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"sdpo.alpha must be in [0, 1], got {self.alpha}.")
        if self.mode not in {"topk", "full", "sampled"}:
            raise ValueError(f"sdpo.mode must be one of topk/full/sampled, got {self.mode!r}.")
        if self.tail not in {"add", "renorm", "drop"}:
            raise ValueError(f"sdpo.tail must be one of add/renorm/drop, got {self.tail!r}.")
        if self.ratio_clip is not None and self.ratio_clip <= 0.0:
            raise ValueError(f"sdpo.ratio_clip must be > 0 when set, got {self.ratio_clip}.")


@dataclass
class DistillationLossConfig(BaseConfig):
    """Configuration for distillation loss settings.

    loss_mode (str):
        Distillation loss function to use.
    topk (int, optional):
        Number of top tokens to consider for top-k distillation losses.
    use_task_rewards (bool):
        Whether to include task rewards alongside distillation loss.
    loss_coef (float):
        Coefficient for distillation loss when combined with task rewards.
    max_clamp (float, optional):
        Maximum value to clamp distillation loss. If None, no clamping is applied.
    log_prob_min_clamp (float, optional):
        Minimum value to clamp log probabilities for stability, e.g., log q - log p where p or q are
        very close to zero. If None, no clamping is applied.
    use_policy_gradient (bool):
        Whether to incorporate distillation loss as a reward, as done
        by https://thinkingmachines.ai/blog/on-policy-distillation/. Recommended to use loss_mode=k1.
        Otherwise, distillation loss is directly backpropagated as a supervised loss,
        as in https://arxiv.org/abs/2306.13649. Recommended to use loss_mode=k3 or forward_kl_topk.
    policy_loss_mode (str):
        Name of the policy loss to use when use_policy_gradient is true.
    clip_ratio (float):
        PPO clipping ratio for policy loss.
    clip_ratio_low (float):
        Lower bound for PPO clipping ratio.
    clip_ratio_high (float):
        Upper bound for PPO clipping ratio.
    loss_settings (DistillationLossSettings, optional):
        Runtime-populated settings based on loss_mode. Not set by user.
    kl_family (str):
        Which set of distillation kernels to use:

        - ``"verl"`` (default): use the existing ``loss_mode`` dispatch
          (``k1/k3/abs/mse/low_var_kl/kl/forward_kl_topk``). All
          ``DistillationLossConfig`` fields above apply.
        - ``"sdpo"``: use the SDPO alpha-parameterized KL kernel. The
          ``sdpo`` sub-config drives behaviour; ``loss_mode`` is auto-set
          to ``"sdpo_alpha_kl_topk"`` (or ``"sdpo_alpha_kl_sampled"`` when
          ``sdpo.mode == "sampled"``) and the legacy verl hyperparameters
          are ignored except for ``max_clamp`` /
          ``log_prob_min_clamp`` / ``loss_coef`` /
          ``use_task_rewards`` / ``use_policy_gradient`` /
          ``policy_loss_mode``.
    sdpo (SDPODistillationLossConfig):
        SDPO-only hyperparameters. Consulted only when
        ``kl_family == "sdpo"``.
    """

    loss_mode: str = "k3"
    topk: Optional[int] = 128
    use_task_rewards: bool = True
    distillation_loss_coef: float = 1.0
    loss_max_clamp: Optional[float] = 10.0
    log_prob_min_clamp: Optional[float] = -10.0

    use_policy_gradient: bool = True
    policy_loss_mode: str = "vanilla"
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2

    kl_family: str = "verl"
    sdpo: "SDPODistillationLossConfig" = field(default_factory=lambda: SDPODistillationLossConfig())

    # ----------------------------------------------------------------------
    # Tricks ported from `revisiting_opd` (https://github.com/hhh675597/revisiting_opd).
    # All default to off; defaults preserve previous behaviour.
    # ----------------------------------------------------------------------
    # Renormalize the top-k slice to sum-to-1 before computing KL inside
    # `reverse_kl_topk`. (For SDPO kernels use `sdpo.tail`.)
    norm_to_one_for_kl: bool = True
    # Clamp (log p_s - log p_t) to [-5, 5] inside the KL kernel for stability.
    # Consumed by `reverse_kl_topk`.
    clip_log_ratio: bool = False
    # Enable head + tail (union-of-K+1) decomposition for `reverse_kl_topk`.
    # The tail term covers the student-sampled token when it lies outside the
    # teacher top-K. Requires `data["sampled_teacher_logprob"]` (alias:
    # `teacher_sampled_logprob`, which OPSD already produces) of shape
    # (B, response_len) — teacher's logprob at the student-sampled token.
    use_tail_sampling: bool = False
    # Multiply the tail term by IW ratio exp(log_pi - log_pi_old).clamp(...)
    # Only consulted when `use_tail_sampling=True`.
    use_kl_iw: bool = False
    kl_iw_clip_lower: Optional[float] = None
    kl_iw_clip_upper: Optional[float] = None
    # Special-token first-occurrence masking (revisiting_opd §4.3): zero out
    # the first occurrence per response of each given token id before loss
    # aggregation. Designed for OPD where student/teacher tokenizers may
    # disagree on chat-template / reasoning markers; for OPSD the vocab is
    # shared so this is a no-op (and a warning is emitted at trainer init).
    opd_mask_special_tokens: bool = False
    opd_mask_first_tokens: list = field(default_factory=lambda: ["<", "think", "<|im_end|>"])
    # Auto-populated at trainer init from `opd_mask_first_tokens` using the
    # trainer tokenizer. Marked mutable in __post_init__.
    opd_mask_token_ids: list = field(default_factory=list)

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    # Store distillation loss settings for computing the specified loss_mode
    # Not set by user, populated at runtime
    loss_settings: Optional[dict] = None

    def __post_init__(self):
        self._mutable_fields.add("loss_settings")
        self._mutable_fields.add("loss_mode")
        self._mutable_fields.add("opd_mask_token_ids")
        from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings

        if self.kl_family not in {"verl", "sdpo"}:
            raise ValueError(
                f"distillation_loss.kl_family must be 'verl' or 'sdpo', got {self.kl_family!r}."
            )
        if self.kl_family == "sdpo":
            # Pin loss_mode to the SDPO kernel matching sdpo.mode. The legacy
            # `loss_mode` field is intentionally overridden here.
            self.loss_mode = (
                "sdpo_alpha_kl_sampled" if self.sdpo.mode == "sampled" else "sdpo_alpha_kl_topk"
            )

        self.loss_settings: DistillationLossSettings = get_distillation_loss_settings(self.loss_mode)

        if self.policy_loss_mode != "vanilla":
            raise NotImplementedError(
                f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                f"but got {self.policy_loss_mode}."
            )

        if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
            print(
                "WARNING: forward_kl_topk is most effective as a supervised distillation loss "
                "(use_policy_gradient=False). With policy gradient, the update uses only the sampled"
                " token's logprob ∇logπ(a), so the top-k distributional signal (how non-sampled logits "
                "should move) is largely unused."
            )

        if not self.use_policy_gradient and self.loss_mode == "k1":
            raise ValueError(
                "Directly backpropagating k1 loss is incorrect since gradient of k1 loss"
                " wrt model weights does not depend on teacher log probabilities."
            )


@dataclass
class DistillationTeacherModelConfig(BaseConfig):
    """Configuration for on-policy distillation teacher.

    key (str, optional):
        Identifier to route examples to the teacher model in multi-teacher setting.
    model_path (str, optional):
        Model path for the teacher model. Can be a local path or a Hugging Face model
    inference (RolloutConfig):
        Rollout configuration for the teacher model inference during distillation.
    num_replicas (int):
        Number of inference replicas of this teacher to launch. Each replica occupies
        `per_replica_world_size` GPUs (= inference.data_parallel_size *
        inference.tensor_model_parallel_size * inference.pipeline_model_parallel_size),
        so the teacher's total GPU footprint is
        `num_replicas * per_replica_world_size`.
    """

    _mutable_fields = BaseConfig._mutable_fields | {"num_replicas", "key"}

    key: Optional[str] = None
    model_path: Optional[str] = None
    inference: RolloutConfig = field(default_factory=RolloutConfig)
    num_replicas: Optional[int] = 0

    @property
    def per_replica_world_size(self) -> int:
        return (
            self.inference.tensor_model_parallel_size
            * self.inference.data_parallel_size
            * self.inference.pipeline_model_parallel_size
        )

    @property
    def world_size(self) -> int:
        return self.num_replicas * self.per_replica_world_size

    def check_configured(self):
        if self.model_path is None:
            raise ValueError("model_path must be specified for distillation teacher model config.")
        if self.key is None:
            raise ValueError("key must be specified for distillation teacher model config.")
        if self.num_replicas is None:
            raise ValueError("num_replicas must be specified for distillation teacher model config.")

    def validate_and_prepare_for_distillation(self, use_topk: bool, topk: Optional[int]) -> None:
        # Prompt + Response from student are fed into teacher as context
        max_model_len = self.inference.max_model_len
        student_prompt_length = self.inference.prompt_length
        student_response_length = self.inference.response_length
        required_context_len = student_prompt_length + student_response_length + 1
        if max_model_len is not None and required_context_len > max_model_len:
            raise ValueError(
                "Distillation teacher inference requires room for the student prompt, the full student "
                f"response, and one generated token, but got {student_prompt_length=}, "
                f"{student_response_length=}, {required_context_len=}, {max_model_len=}."
            )
        self.inference.prompt_length = self.inference.prompt_length + self.inference.response_length
        self.inference.response_length = 1
        self._validate_topk_logprobs(use_topk=use_topk, topk=topk)

    def _validate_topk_logprobs(self, use_topk: bool, topk: Optional[int]) -> None:
        if not use_topk:
            return
        if topk is None:
            raise ValueError("topk must be specified when use_topk is True.")

        engine_name = self.inference.name
        engine_kwargs = self.inference.engine_kwargs
        match engine_name:
            case "vllm":
                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
                if max_logprobs is None:
                    vllm_engine_kwargs["max_logprobs"] = topk
                    max_logprobs = topk
                if max_logprobs < topk:
                    raise ValueError(
                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                        f"({topk}) to enable distillation loss computation."
                    )
                engine_kwargs["vllm"] = vllm_engine_kwargs
            case "sglang":
                # SGLang's top_logprobs_num is a per-request parameter, so there is no
                # engine-boot cap to align (unlike vLLM's max_logprobs). The async
                # server translates sampling_params["prompt_logprobs"] into
                # return_logprob + logprob_start_len=0 + top_logprobs_num at call time.
                pass
            case _:
                raise NotImplementedError(
                    f"DistillationTeacherModelConfig does not support inference engine {engine_name}"
                )


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for on-policy distillation.

    enabled (bool):
        Whether on-policy distillation is enabled.
    n_gpus_per_node (int):
        Number of GPUs per node in the teacher resource pool.
    nnodes (int):
        Number of nodes in the teacher resource pool.
    teacher_models (dict[str, TeacherModelConfig]):
        Configurations for teacher models used for multi-teacher distillation.
    teacher_key (str):
        Key to route examples to the appropriate teacher model in multi-teacher setups. Should correspond to a field in
        the data proto, e.g., data_source.
    distillation_loss (DistillationLossConfig):
    Configuration for distillation loss settings.

    NOTE: The `teacher_model` entry is in the `teacher_models` dict by default.
    Since it is popped when other teacher entries are added, using `teacher_model` as
    one of several keys silently drops it. For example, the following CLI overrides result
    in ONLY `teacher_model2` being used:

    ```bash
    distillation.teacher_models.teacher_model.key=openai/gsm8k
    distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    Instead, give the first teacher a different name:

    ```bash
    +distillation.teacher_models.teacher_model1.key=openai/gsm8k
    +distillation.teacher_models.teacher_model1.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    """

    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models", "n_gpus_per_node", "nnodes"}

    enabled: bool = False
    mode: str = "external"
    n_gpus_per_node: int = 0
    nnodes: int = 0
    teacher_models: dict[str, DistillationTeacherModelConfig] = field(default_factory=dict)
    teacher_key: str = "data_source"
    self_distill: "SelfDistillationConfig" = field(default_factory=lambda: SelfDistillationConfig())
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)

    def __post_init__(self):
        if not self.enabled:
            return

        if self.mode not in {"external", "self"}:
            raise ValueError(f"distillation.mode must be 'external' or 'self', got {self.mode!r}.")

        if self.mode == "self":
            # OPSD: teacher is colocated with the actor; no extra teacher pool needed.
            self.self_distill.validate()
            return

        # external mode: OPSD-only knobs must be left at defaults; the
        # teacher is an out-of-process model and cannot be updated via
        # ema / progressive / trust_region.
        self.self_distill.assert_unused_for_external()

        self.teacher_models = self._resolve_teacher_models()
        teacher_world_size_sum = 0
        for teacher_model in self.teacher_models.values():
            teacher_model.validate_and_prepare_for_distillation(
                use_topk=self.distillation_loss.loss_settings.use_topk,
                topk=self.distillation_loss.topk,
            )
            teacher_world_size_sum += teacher_model.world_size
        total_pool_size = self.n_gpus_per_node * self.nnodes
        if teacher_world_size_sum != total_pool_size:
            raise ValueError(
                f"Sum of teacher (num_replicas * per_replica_world_size) ({teacher_world_size_sum}) must match "
                f"the distillation resource pool size "
                f"({self.n_gpus_per_node=} * {self.nnodes=} = {total_pool_size})."
            )

    def _resolve_teacher_models(self) -> dict[str, DistillationTeacherModelConfig]:
        assert "teacher_model" in self.teacher_models
        if len(self.teacher_models) == 1:
            # Single teacher occupies the entire teacher resource pool.
            teacher_model = self.teacher_models["teacher_model"]
            inference = teacher_model.inference
            per_replica = (
                inference.tensor_model_parallel_size
                * inference.data_parallel_size
                * inference.pipeline_model_parallel_size
            )
            pool_size = self.n_gpus_per_node * self.nnodes
            if pool_size % per_replica != 0:
                raise ValueError(
                    f"Single teacher's per_replica_world_size ({per_replica}) must divide the distillation "
                    f"resource pool size ({self.n_gpus_per_node=} * {self.nnodes=} = {pool_size})."
                )
            teacher_model.num_replicas = pool_size // per_replica
            teacher_model.key = "default"
        else:
            # Multiple teachers: remove default single teacher config
            self.teacher_models.pop("teacher_model")

        # Teacher models dict is keyed by teacher_key instead of YAML entry name
        teacher_models = {}
        for teacher_config in self.teacher_models.values():
            teacher_config = omega_conf_to_dataclass(teacher_config, dataclass_type=DistillationTeacherModelConfig)
            teacher_config.check_configured()
            if teacher_config.key in teacher_models:
                raise ValueError(f"Duplicate teacher key {teacher_config.key} found in teacher models.")
            teacher_models[teacher_config.key] = teacher_config
        return teacher_models


@dataclass
class SelfDistillationConfig(BaseConfig):
    """OPSD-only settings (consulted only when ``distillation.mode == "self"``).

    teacher_update (str):
        How the teacher policy is obtained from the student. One of:

        - ``"ref"``: keep the teacher frozen at the initial reference model
          (vanilla on-policy distillation against a fixed teacher).
        - ``"ema"`` (default): every step do
          ``ref ← ema_decay·ref + (1-ema_decay)·actor``. The teacher tracks
          the student smoothly and acts as a moving target / soft trust
          region; empirically the most stable choice for self-distillation.
        - ``"progressive"``: every ``teacher_update_interval`` steps hard-copy
          ``ref ← actor`` (delayed self-teacher snapshot).
        - ``"trust_region"``: hard-copy ``ref ← actor`` only when the
          student-vs-teacher KL falls below ``trust_region_threshold`` (i.e.
          the student has "caught up" enough that the teacher should advance).
    ema_decay (float):
        EMA decay used when ``teacher_update == "ema"``. Typical
        values: 0.99 – 0.9999.
    teacher_update_interval (int):
        Step interval between teacher snapshots in ``"progressive"`` mode
        (must be > 0 for that mode).
    trust_region_threshold (float):
        KL threshold under which the teacher snapshot is refreshed in
        ``"trust_region"`` mode. Lower values → more conservative updates.
        Set to 0.0 to refresh every step (equivalent to ``"progressive"``
        with interval 1).
    teacher_update_rate (float):
        Deprecated alias kept for backward compatibility; if non-zero and
        ``teacher_update == "ema"``, takes precedence over
        ``ema_decay`` as ``1 - teacher_update_rate``.
    dataloader (str):
        Importable FQN of a subclass of
        :class:`verl.workers.self_distillation.OfflineTeacherDataloader` or
        :class:`verl.workers.self_distillation.OnlineTeacherDataloader`,
        e.g. ``"my_pkg.my_module:MyTeacherDataloader"``. Required.
        Offline vs online is auto-detected at load time via ``isinstance``.
    dataloader_kwargs (dict):
        Forwarded as keyword arguments to the dataloader class's ``__init__``.
        ``tokenizer`` and ``processor`` are injected by the framework.
    truncation (str):
        Truncation strategy when the teacher prompt exceeds
        ``data.max_prompt_length``: ``"left"`` / ``"right"`` / ``"error"``.
    sample_dump_path (str, optional):
        Directory for debugging per-sample OPSD inputs/outputs. Each step is
        written as ``<global_step>.jsonl`` with first_100 response-token
        student/teacher logprob traces. Disabled when ``None``.
    sample_dump_max_per_step (int):
        Maximum number of samples to dump per training step. ``0`` means dump
        the full batch.
    """

    teacher_update: str = "ema"
    ema_decay: float = 0.999
    teacher_update_rate: float = 0.0
    teacher_update_interval: int = 0
    trust_region_threshold: float = 0.0
    dataloader: Optional[str] = None
    dataloader_kwargs: dict = field(default_factory=dict)
    truncation: str = "right"
    sample_dump_path: Optional[str] = None
    sample_dump_max_per_step: int = 0

    def validate(self) -> None:
        if self.teacher_update not in {"ref", "ema", "trust_region", "progressive"}:
            raise ValueError(
                f"teacher_update must be one of ref/ema/trust_region/progressive, "
                f"got {self.teacher_update!r}."
            )
        if self.teacher_update == "ema":
            decay = (
                1.0 - self.teacher_update_rate
                if self.teacher_update_rate > 0.0
                else self.ema_decay
            )
            if not (0.0 < decay < 1.0):
                raise ValueError(
                    f"OPSD ema teacher requires 0 < ema_decay < 1, got {decay}."
                )
        if self.teacher_update == "progressive" and self.teacher_update_interval <= 0:
            raise ValueError(
                "OPSD progressive teacher requires teacher_update_interval > 0."
            )
        if self.teacher_update == "trust_region" and self.trust_region_threshold < 0.0:
            raise ValueError(
                "OPSD trust_region teacher requires trust_region_threshold >= 0."
            )
        if not self.dataloader:
            raise ValueError(
                "OPSD requires distillation.self_distill.dataloader to be set."
            )
        if self.sample_dump_max_per_step < 0:
            raise ValueError(
                "OPSD sample_dump_max_per_step must be >= 0."
            )

    def assert_unused_for_external(self) -> None:
        """Reject OPSD-only overrides when distillation.mode != 'self'.

        ``ema`` / ``progressive`` / ``trust_region`` teacher updates and the
        custom self-distillation dataloader only make sense when the teacher
        is the colocated ref policy. In external-teacher mode the teacher is
        an out-of-process model that the trainer cannot mutate, so these
        knobs would silently no-op; we raise to make the misconfiguration
        explicit. Detection is "did the user touch any OPSD-only field"
        rather than the value of ``teacher_update`` alone, so that plain
        external runs that never set ``self_distill.*`` keep working.
        """
        offenders: list[str] = []
        if self.teacher_update not in {"ref", "ema"}:
            offenders.append(f"teacher_update={self.teacher_update!r}")
        if self.teacher_update_rate != 0.0:
            offenders.append(f"teacher_update_rate={self.teacher_update_rate}")
        if self.teacher_update_interval != 0:
            offenders.append(f"teacher_update_interval={self.teacher_update_interval}")
        if self.trust_region_threshold != 0.0:
            offenders.append(f"trust_region_threshold={self.trust_region_threshold}")
        if self.ema_decay != 0.999:
            offenders.append(f"ema_decay={self.ema_decay}")
        if self.dataloader:
            offenders.append(f"dataloader={self.dataloader!r}")
        if self.dataloader_kwargs:
            offenders.append(f"dataloader_kwargs={self.dataloader_kwargs!r}")
        if self.sample_dump_path is not None:
            offenders.append(f"sample_dump_path={self.sample_dump_path!r}")
        if self.sample_dump_max_per_step != 0:
            offenders.append(f"sample_dump_max_per_step={self.sample_dump_max_per_step}")
        if offenders:
            raise ValueError(
                "distillation.self_distill.* is only supported when "
                "distillation.mode == 'self' (OPSD). The following overrides "
                f"are not applicable to external-teacher distillation: {', '.join(offenders)}. "
                "Set distillation.mode=self or remove these fields."
            )
