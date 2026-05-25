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


import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    return {"distillation_losses": distillation_losses, "student_mass": student_mass, "teacher_mass": teacher_mass}


def _add_tail_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """Append a virtual "tail" bucket carrying the residual mass log(1 - Σ p_i).

    Uses ``log(1 - exp(log_s)) = log(-expm1(log_s))`` for numerical stability,
    clamping ``log_s`` slightly below 0 to avoid log(0) when the top-k mass
    already saturates the distribution. Returns a tensor with shape
    ``(..., topk + 1)``.
    """
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = log_s.clamp(max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


def _renorm_topk_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """Re-normalise top-k log-probs so they sum to 1."""
    return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)


def compute_sdpo_alpha_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """SDPO-style alpha-parameterised KL on top-k log-probabilities.

    alpha == 0 → forward KL, alpha == 1 → reverse KL, alpha == 0.5 → JSD,
    alpha ∈ (0,1) \\ {0.5} → skew / generalised JSD. Mirrors the kernel in
    SDPO core_algos (`verl/trainer/ppo/core_algos.py:1138-1163`).
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)

    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)

    sdpo_cfg = loss_config.sdpo
    student_dist = student_topk_log_probs.float()
    teacher_dist = teacher_topk_log_probs.float()
    if sdpo_cfg.tail == "add":
        student_dist = _add_tail_log_probs(student_dist)
        teacher_dist = _add_tail_log_probs(teacher_dist)
    elif sdpo_cfg.tail == "renorm":
        student_dist = _renorm_topk_log_probs(student_dist)
        teacher_dist = _renorm_topk_log_probs(teacher_dist)
    # tail == "drop": use raw top-K log-probs unchanged

    alpha = float(sdpo_cfg.alpha)
    if alpha == 0.0:
        # forward KL: KL(teacher || student) = Σ p_t (log p_t - log p_s)
        per_pos = F.kl_div(student_dist, teacher_dist, reduction="none", log_target=True)
    elif alpha == 1.0:
        # reverse KL: KL(student || teacher) = Σ p_s (log p_s - log p_t)
        per_pos = F.kl_div(teacher_dist, student_dist, reduction="none", log_target=True)
    else:
        # (skew) Jensen-Shannon divergence:
        # m = (1-α)·p_s + α·p_t  →  loss = (1-α)·KL(m||p_s) + α·KL(m||p_t)
        alpha_t = torch.tensor(alpha, dtype=student_dist.dtype, device=student_dist.device)
        log_mix = torch.logsumexp(
            torch.stack(
                [student_dist + torch.log(1.0 - alpha_t), teacher_dist + torch.log(alpha_t)]
            ),
            dim=0,
        )
        kl_to_student = F.kl_div(log_mix, student_dist, reduction="none", log_target=True)
        kl_to_teacher = F.kl_div(log_mix, teacher_dist, reduction="none", log_target=True)
        per_pos = torch.lerp(kl_to_student, kl_to_teacher, alpha_t)

    distillation_losses = per_pos.sum(dim=-1)
    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
    }


def compute_reverse_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
    sampled_token_ids: torch.Tensor | None = None,
    sampled_teacher_logprob: torch.Tensor | None = None,
) -> dict:
    """Reverse KL over teacher top-K, optionally with union-of-(K+1) tail term.

    Head term L1 = KL(pi_student || pi_teacher) restricted to teacher's top-K.
    Tail term L2 covers the student-sampled token when it lies outside the
    top-K (zero otherwise), enabled by ``loss_config.use_tail_sampling``.
    See revisiting_opd `compute_memory_efficient_kl`.
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)

    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs_c = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs_c = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    else:
        student_topk_log_probs_c = student_topk_log_probs
        teacher_topk_log_probs_c = teacher_topk_log_probs

    norm_to_one = bool(getattr(loss_config, "norm_to_one_for_kl", True))
    clip_ratio = bool(getattr(loss_config, "clip_log_ratio", False))
    use_tail = bool(getattr(loss_config, "use_tail_sampling", False))

    if use_tail:
        if sampled_token_ids is None or sampled_teacher_logprob is None:
            raise ValueError(
                "reverse_kl_topk: use_tail_sampling=True requires data['responses'] and "
                "data['sampled_teacher_logprob'] (or 'teacher_sampled_logprob') of shape "
                "(B, response_len)."
            )
        sampled_token_ids_local = sampled_token_ids
        sampled_teacher_logprob_local = sampled_teacher_logprob
        if sampled_token_ids_local.dim() == 2:
            sampled_token_ids_local = sampled_token_ids_local.reshape(1, -1)
            sampled_teacher_logprob_local = sampled_teacher_logprob_local.reshape(1, -1)
        if get_ulysses_sequence_parallel_world_size() > 1:
            sampled_token_ids_local = slice_input_tensor(sampled_token_ids_local, dim=1)
            sampled_teacher_logprob_local = slice_input_tensor(sampled_teacher_logprob_local, dim=1)

        in_topk = (teacher_topk_ids == sampled_token_ids_local.unsqueeze(-1)).any(dim=-1)
        not_in_topk = ~in_topk
        not_in_topk_ratio = not_in_topk.float().mean()

        student_sampled_logprob = torch.gather(
            student_log_probs, dim=-1, index=sampled_token_ids_local.unsqueeze(-1).long()
        ).squeeze(-1)
        if loss_config.log_prob_min_clamp is not None:
            student_sampled_logprob_c = student_sampled_logprob.clamp_min(loss_config.log_prob_min_clamp)
            teacher_sampled_logprob_c = sampled_teacher_logprob_local.clamp_min(loss_config.log_prob_min_clamp)
        else:
            student_sampled_logprob_c = student_sampled_logprob
            teacher_sampled_logprob_c = sampled_teacher_logprob_local

        # Pad in-topk slots with -inf so exp() zeros them; gates L2 automatically.
        neg_inf = torch.finfo(student_topk_log_probs_c.dtype).min
        s_extra = torch.where(not_in_topk, student_sampled_logprob_c, torch.full_like(student_sampled_logprob_c, neg_inf))
        t_extra = torch.where(not_in_topk, teacher_sampled_logprob_c, torch.full_like(teacher_sampled_logprob_c, neg_inf))
        s_union = torch.cat([student_topk_log_probs_c, s_extra.unsqueeze(-1)], dim=-1)
        t_union = torch.cat([teacher_topk_log_probs_c, t_extra.unsqueeze(-1)], dim=-1)

        if norm_to_one:
            s = F.log_softmax(s_union, dim=-1)
            t = F.log_softmax(t_union, dim=-1)
        else:
            s = s_union
            t = t_union
        diff = s - t
        if clip_ratio:
            diff = diff.clamp(min=-5.0, max=5.0)
        weighted = s.exp() * diff
        L1 = weighted[..., :-1].sum(dim=-1)
        L2 = weighted[..., -1]
    else:
        if norm_to_one:
            s = F.log_softmax(student_topk_log_probs_c, dim=-1)
            t = F.log_softmax(teacher_topk_log_probs_c, dim=-1)
        else:
            s = student_topk_log_probs_c
            t = teacher_topk_log_probs_c
        diff = s - t
        if clip_ratio:
            diff = diff.clamp(min=-5.0, max=5.0)
        L1 = (s.exp() * diff).sum(dim=-1)
        L2 = torch.zeros_like(L1)
        not_in_topk_ratio = torch.zeros((), device=L1.device, dtype=L1.dtype)

    return {
        "distillation_losses": L1 + L2,
        "distillation_losses_head": L1,
        "distillation_losses_tail": L2,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "not_in_topk_ratio": not_in_topk_ratio,
    }