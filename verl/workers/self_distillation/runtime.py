# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Runtime glue between OPSD teacher dataloaders and verl's batch pipeline.

For each sample in a ``DataProto``:
  1. Decode the student prompt (and response, for online dataloaders).
  2. Call the user-supplied dataloader's ``build_one``.
  3. Re-apply the **same** tokenizer / multi-modal processor used by the
     student to the returned chat messages.
  4. Concatenate the (unchanged) student response token IDs to form the
     teacher's full input sequence.
  5. Write the following batch fields:
     * ``teacher_input_ids`` / ``teacher_attention_mask``           (left-padded prompt)
     * ``teacher_full_input_ids`` / ``teacher_full_attention_mask`` (prompt ⊕ response)
     * ``teacher_full_position_ids``
     * ``self_distillation_mask`` — per-sample (B,) bool; ``False`` = skip
     * ``teacher_multi_modal_inputs`` (non-tensor, only when populated) — list
       of per-sample dicts of processor outputs minus ``input_ids`` /
       ``attention_mask`` / ``mm_token_type_ids``; key matches the engine's
       ``extract_multi_modal_inputs`` convention.

The teacher *logprobs* themselves are produced later by a forward pass over
``teacher_full_input_ids`` — see ``ray_trainer._compute_self_distill_teacher_log_prob``.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Optional

import numpy as np
import torch

from .teacher_dataloader import (
    BatchView,
    OfflineTeacherDataloader,
    OnlineTeacherDataloader,
    TeacherDataloader,
    TeacherSample,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def load_teacher_dataloader_from_spec(
    spec: dict | None,
    *,
    tokenizer,
    processor=None,
) -> Optional[TeacherDataloader]:
    """Instantiate a dataloader from a YAML/Hydra block ``{target, kwargs}``.

    ``target`` is either:
      * a short name registered via
        :func:`verl.workers.self_distillation.dataloaders.register`
        (e.g. ``"vision_opd"``, ``"opsd"``, ``"sdpo"``), or
      * an importable FQN ``pkg.module:Class`` / ``pkg.module.Class`` of a
        :class:`OfflineTeacherDataloader` / :class:`OnlineTeacherDataloader`
        subclass (for out-of-tree user code).

    Offline vs online is **auto-detected** at load time via ``isinstance``.
    """
    if not spec:
        return None
    target: Optional[str] = spec.get("target")
    if not target:
        return None
    kwargs: dict = dict(spec.get("kwargs") or {})

    # Short-name registry lookup first.
    from .dataloaders import get as _registry_get

    cls = _registry_get(target)
    if cls is None:
        mod_name, _, cls_name = target.replace(":", ".").rpartition(".")
        if not mod_name or not cls_name:
            raise ValueError(
                f"Invalid teacher_dataloader target {target!r}; expected a registered short name "
                f"or 'pkg.module:Class' FQN."
            )
        cls = getattr(importlib.import_module(mod_name), cls_name)
    if not issubclass(cls, TeacherDataloader):
        raise TypeError(
            f"{target} is not a subclass of OfflineTeacherDataloader or OnlineTeacherDataloader."
        )
    return cls(tokenizer=tokenizer, processor=processor, **kwargs)


class SelfDistillationRuntime:
    """Apply one teacher dataloader to a batch.

    The runtime intentionally knows nothing about FSDP/Ray/loss; it only
    manipulates per-sample CPU tensors so it can be re-used from any worker.
    """

    def __init__(
        self,
        *,
        teacher_dataloader: TeacherDataloader,
        tokenizer,
        processor=None,
        max_prompt_length: int,
        pad_token_id: int,
        truncation: str = "right",
    ):
        self.dataloader = teacher_dataloader
        self.is_online = isinstance(teacher_dataloader, OnlineTeacherDataloader)
        self.is_offline = isinstance(teacher_dataloader, OfflineTeacherDataloader)
        if self.is_online == self.is_offline:
            raise TypeError(
                f"{type(teacher_dataloader).__name__} must subclass exactly one of "
                f"OfflineTeacherDataloader / OnlineTeacherDataloader."
            )
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.pad_token_id = pad_token_id
        if truncation not in {"left", "right", "error"}:
            raise ValueError(f"truncation must be left/right/error, got {truncation!r}")
        self.truncation = truncation

    @property
    def mode(self) -> str:
        return "online" if self.is_online else "offline"

    # ---------------------------------------------------------------- public

    def apply(self, batch) -> None:
        """In-place mutate ``batch`` (a verl ``DataProto``) with teacher-side fields.

        For VL models with ``processor.get_rope_index`` (Qwen2.5/3.5-VL, GLM4V,
        etc.) the per-sample prompt position_ids are computed in MROPE form
        (3 channels) mirroring the reference Vision-OPD
        ``_build_teacher_prompt_inputs``; otherwise we fall back to the 1D
        ``cumsum(attn) - 1`` form.
        """
        n = batch.batch.batch_size[0]
        prompt_ids: torch.Tensor = batch.batch["prompts"]
        response_ids: torch.Tensor = batch.batch["responses"]

        view = self._build_batch_view(batch, n) if self.is_online else None

        teacher_samples: list[TeacherSample] = [
            self._call_dataloader(batch, i, view) for i in range(n)
        ]
        skip_mask = torch.tensor([s.skip for s in teacher_samples], dtype=torch.bool)

        prompt_ids_list: list[torch.Tensor] = []
        prompt_attn_list: list[torch.Tensor] = []
        prompt_pos_list: list[Optional[torch.Tensor]] = []
        mm_inputs_per_sample: list[Optional[dict]] = []

        for i, sample in enumerate(teacher_samples):
            if sample.skip or not sample.messages:
                tp_ids = self._extract_unpadded_prompt(prompt_ids[i])
                attn = torch.ones_like(tp_ids, dtype=torch.long)
                prompt_ids_list.append(tp_ids)
                prompt_attn_list.append(attn)
                prompt_pos_list.append(None)
                mm_inputs_per_sample.append(None)
                continue
            tp_ids, attn, mm, pos = self._tokenize_messages(sample)
            prompt_ids_list.append(tp_ids)
            prompt_attn_list.append(attn)
            prompt_pos_list.append(pos)
            mm_inputs_per_sample.append(mm)

        teacher_prompt_ids, teacher_prompt_mask = self._left_pad(
            prompt_ids_list, prompt_attn_list, self.pad_token_id
        )
        teacher_full_ids = torch.cat([teacher_prompt_ids, response_ids], dim=1)

        # response-side mask: reuse student's if available
        full_attn_student = batch.batch.get("attention_mask")
        tr = response_ids.shape[1]
        if full_attn_student is not None:
            resp_mask = full_attn_student[:, -tr:].long()
        else:
            resp_mask = (response_ids != self.pad_token_id).long()
        teacher_full_attn = torch.cat([teacher_prompt_mask, resp_mask], dim=1)

        teacher_full_position_ids = self._assemble_full_position_ids(
            prompt_pos_list,
            teacher_prompt_ids,
            teacher_prompt_mask,
            teacher_full_attn,
            tr,
        )

        batch.batch["teacher_input_ids"] = teacher_prompt_ids
        batch.batch["teacher_attention_mask"] = teacher_prompt_mask
        batch.batch["teacher_full_input_ids"] = teacher_full_ids
        batch.batch["teacher_full_attention_mask"] = teacher_full_attn
        batch.batch["teacher_full_position_ids"] = teacher_full_position_ids
        batch.batch["self_distillation_mask"] = ~skip_mask

        # Write under the engine-consumed key (``multi_modal_inputs``) on a
        # *teacher_*-prefixed slot so callers can route it explicitly into the
        # ref forward without clobbering the student copy. ``mm_token_type_ids``
        # is already stripped in ``_tokenize_messages`` so per-sample tensors
        # are safe to ``torch.cat`` on dim 0 inside ``extract_multi_modal_inputs``.
        if any(m is not None for m in mm_inputs_per_sample):
            batch.non_tensor_batch["teacher_multi_modal_inputs"] = np.array(
                [m if m is not None else {} for m in mm_inputs_per_sample],
                dtype=object,
            )

    # --------------------------------------------------------------- private

    def _build_batch_view(self, batch, n: int) -> BatchView:
        uids = list(batch.non_tensor_batch.get("uid", [None] * n))

        rewards = None
        if "token_level_rewards" in batch.batch:
            r = batch.batch["token_level_rewards"]
            rewards = r.sum(dim=-1).tolist() if r.ndim > 1 else r.tolist()
        elif "reward_score" in batch.batch:
            rewards = batch.batch["reward_score"].tolist()

        response_texts = self.tokenizer.batch_decode(
            batch.batch["responses"], skip_special_tokens=True
        )

        return BatchView(
            uids=uids,
            rewards=rewards,
            response_texts=response_texts,
            extra={k: list(v) for k, v in batch.non_tensor_batch.items()},
        )

    def _call_dataloader(self, batch, i: int, view: Optional[BatchView]) -> TeacherSample:
        prompt_ids: torch.Tensor = batch.batch["prompts"][i]
        prompt_text = self.tokenizer.decode(
            prompt_ids[prompt_ids != self.pad_token_id], skip_special_tokens=False
        )

        prompt_messages = None
        raw_messages = batch.non_tensor_batch.get("raw_prompt")
        if raw_messages is not None and raw_messages[i] is not None:
            prompt_messages = list(raw_messages[i])

        mm_data = None
        if "multi_modal_data" in batch.non_tensor_batch:
            mm_data = batch.non_tensor_batch["multi_modal_data"][i]

        extra_info = batch.non_tensor_batch.get("extra_info")
        extra_info_i = (
            dict(extra_info[i])
            if extra_info is not None and extra_info[i] is not None
            else {}
        )

        if self.is_offline:
            return self.dataloader.build_one(
                prompt_messages=prompt_messages,
                prompt_text=prompt_text,
                multi_modal_data=mm_data,
                extra_info=extra_info_i,
            )
        assert view is not None
        response_text = view.response_texts[i] if view.response_texts else ""
        reward = view.rewards[i] if view.rewards is not None else None
        return self.dataloader.build_one(
            prompt_messages=prompt_messages,
            prompt_text=prompt_text,
            response_text=response_text,
            reward=reward,
            multi_modal_data=mm_data,
            extra_info=extra_info_i,
            batch_view=view,
            index=i,
        )

    def _tokenize_messages(self, sample: TeacherSample) -> tuple[torch.Tensor, torch.Tensor, Optional[dict], Optional[torch.Tensor]]:
        """Re-encode teacher messages and (for VL models) compute MROPE position_ids.

        Returns ``(input_ids, attention_mask, multi_modal_inputs, position_ids)``.
        ``position_ids`` is ``None`` for text-only / no-rope models; for VL it
        is a ``(rope_dims, T_prompt)`` tensor computed via
        ``processor.get_rope_index`` — same call site as
        ``RayPPOTrainer._build_teacher_prompt_inputs`` in the reference.
        ``mm_token_type_ids`` is consumed here for rope and then dropped from
        the returned mm dict (the engine path discards it: it is only used to
        compute position ids).
        """
        if self.processor is not None and sample.multi_modal_data:
            # NOTE: keys mirror verl's AgentLoop / RLHFDataset convention
            # (plural: "images" / "videos" / "audios"), not the HF processor's
            # singular kwargs.
            text = self.processor.apply_chat_template(
                sample.messages, tokenize=False, add_generation_prompt=True
            )
            mm_inputs = self.processor(
                text=[text],
                images=sample.multi_modal_data.get("images"),
                videos=sample.multi_modal_data.get("videos"),
                return_tensors="pt",
                add_special_tokens=False,
            )
            ids = mm_inputs["input_ids"][0]
            attn = mm_inputs["attention_mask"][0]
            mm_inputs_out = {
                k: v for k, v in mm_inputs.items() if k not in {"input_ids", "attention_mask"}
            }
            pos = self._compute_prompt_position_ids(ids, attn, mm_inputs_out)
            mm_inputs_out.pop("mm_token_type_ids", None)
        else:
            text = self.tokenizer.apply_chat_template(
                sample.messages, tokenize=False, add_generation_prompt=True
            )
            enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"][0]
            attn = enc["attention_mask"][0]
            mm_inputs_out = sample.multi_modal_data
            pos = None

        ids, attn = self._maybe_truncate(ids, attn)
        if pos is not None and pos.shape[-1] > ids.shape[0]:
            pos = pos[..., : ids.shape[0]] if self.truncation != "left" else pos[..., -ids.shape[0] :]
        return ids.long(), attn.long(), mm_inputs_out, pos

    def _compute_prompt_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mm_inputs: dict,
    ) -> Optional[torch.Tensor]:
        """Run ``processor.get_rope_index`` for the teacher prompt.

        Mirrors ``verl/experimental/agent_loop/agent_loop.py::_compute_position_ids``
        and ``Vision-OPD/.../ray_trainer.py::_build_teacher_prompt_inputs``:
        ``mm_token_type_ids`` is popped (rebuilt from input_ids if absent) and
        fed to ``get_rope_index`` alongside ``image_grid_thw`` / ``video_grid_thw``.
        Returns a ``(rope_dims, T)`` tensor, or ``None`` if the processor has no
        ``get_rope_index`` (text-only / model not VL).
        """
        if self.processor is None or not hasattr(self.processor, "get_rope_index"):
            return None
        try:
            ids = input_ids.unsqueeze(0)
            attn = attention_mask.unsqueeze(0)
            mm_token_type_ids = mm_inputs.get("mm_token_type_ids")
            if mm_token_type_ids is None:
                image_token_id = getattr(self.processor, "image_token_id", None)
                video_token_id = getattr(self.processor, "video_token_id", None)
                if image_token_id is None and video_token_id is None:
                    return None
                mm_token_type_ids = torch.zeros_like(ids)
                if image_token_id is not None:
                    mm_token_type_ids[0][ids[0] == int(image_token_id)] = 1
                if video_token_id is not None:
                    mm_token_type_ids[0][ids[0] == int(video_token_id)] = 2

            kwargs = dict(
                input_ids=ids,
                attention_mask=attn,
                image_grid_thw=mm_inputs.get("image_grid_thw"),
                video_grid_thw=mm_inputs.get("video_grid_thw"),
            )
            # Newer Qwen3.5-VL / Qwen3-VL get_rope_index signature accepts
            # ``mm_token_type_ids``; older qwen2_vl-style signatures do not.
            import inspect
            try:
                sig_params = inspect.signature(self.processor.get_rope_index).parameters
            except (TypeError, ValueError):
                sig_params = {}
            if "mm_token_type_ids" in sig_params:
                kwargs["mm_token_type_ids"] = mm_token_type_ids
            if "second_per_grid_ts" in sig_params:
                kwargs["second_per_grid_ts"] = mm_inputs.get("second_per_grid_ts")

            out = self.processor.get_rope_index(**kwargs)
            if isinstance(out, tuple):
                out = out[0]
            # Normalize to (rope_dims, T).
            if out.dim() == 3:
                # (rope_dims, B, T) → (rope_dims, T) since B == 1
                out = out[:, 0, :]
            elif out.dim() == 2 and out.shape[0] == 1:
                # (1, T) — text rope; squeeze
                out = out[0]
            return out.long()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "teacher rope_index computation failed (%s); falling back to 1D cumsum.",
                exc,
            )
            return None

    def _assemble_full_position_ids(
        self,
        prompt_pos_list: list[Optional[torch.Tensor]],
        teacher_prompt_ids: torch.Tensor,
        teacher_prompt_mask: torch.Tensor,
        teacher_full_attn: torch.Tensor,
        T_resp: int,
    ) -> torch.Tensor:
        """Build ``(B, [rope_dims,] T_full)`` position_ids.

        If any sample produced N-D mrope positions, output is ``(B, rope_dims, T_full)``.
        Otherwise output is the 1D ``cumsum(attn) - 1`` form ``(B, T_full)``.
        Response-side ids extend linearly from each channel's max prompt id.
        """
        B = teacher_prompt_ids.shape[0]
        T_prompt = teacher_prompt_ids.shape[1]
        T_full = T_prompt + T_resp
        device = teacher_prompt_ids.device

        rope_dims = 0
        for p in prompt_pos_list:
            if p is not None and p.dim() == 2:
                rope_dims = max(rope_dims, p.shape[0])

        if rope_dims == 0:
            return (teacher_full_attn.cumsum(dim=-1) - 1).clamp(min=0).long()

        out = torch.zeros((B, rope_dims, T_full), dtype=torch.long, device=device)
        for i in range(B):
            attn_i = teacher_prompt_mask[i]
            valid_len = int(attn_i.sum().item())
            prompt_pos = prompt_pos_list[i]
            if prompt_pos is None:
                # Skipped or text-only sample — broadcast 1D cumsum across all rope dims.
                pos_1d = (teacher_full_attn[i].cumsum(dim=-1) - 1).clamp(min=0).long()
                out[i] = pos_1d.unsqueeze(0).expand(rope_dims, -1)
                continue
            if prompt_pos.dim() == 1:
                prompt_pos = prompt_pos.unsqueeze(0).expand(rope_dims, -1).contiguous()
            # Left-pad each channel to T_prompt, then append linearly-advancing
            # response positions.
            pad_l = T_prompt - prompt_pos.shape[-1]
            for k in range(rope_dims):
                pos_k = prompt_pos[k].to(device)
                if pad_l > 0:
                    # Pads are inactive — give them id 0; valid section is right-aligned.
                    out[i, k, pad_l : pad_l + pos_k.shape[0]] = pos_k
                else:
                    out[i, k, :T_prompt] = pos_k[:T_prompt]
                max_pos = int(out[i, k, :T_prompt].max().item()) if valid_len > 0 else -1
                # Response tokens (no padding on the right): id = max_pos + 1 + arange.
                out[i, k, T_prompt:] = torch.arange(
                    max_pos + 1, max_pos + 1 + T_resp, dtype=torch.long, device=device
                )
        return out

    def _maybe_truncate(self, ids: torch.Tensor, attn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if ids.shape[0] <= self.max_prompt_length:
            return ids, attn
        if self.truncation == "error":
            raise ValueError(
                f"Teacher prompt too long: {ids.shape[0]} > max_prompt_length={self.max_prompt_length}"
            )
        if self.truncation == "left":
            return ids[-self.max_prompt_length :], attn[-self.max_prompt_length :]
        return ids[: self.max_prompt_length], attn[: self.max_prompt_length]

    def _extract_unpadded_prompt(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        mask = prompt_ids != self.pad_token_id
        if mask.any():
            first = int(mask.float().argmax().item())
            return prompt_ids[first:]
        return prompt_ids

    def _left_pad(
        self,
        ids_list: list[torch.Tensor],
        mask_list: list[torch.Tensor],
        pad_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = min(max(t.shape[0] for t in ids_list), self.max_prompt_length)
        out_ids, out_attn = [], []
        for ids, attn in zip(ids_list, mask_list):
            if ids.shape[0] > max_len:
                if self.truncation == "left":
                    ids, attn = ids[-max_len:], attn[-max_len:]
                else:
                    ids, attn = ids[:max_len], attn[:max_len]
            pad = max_len - ids.shape[0]
            if pad > 0:
                ids = torch.cat([torch.full((pad,), pad_id, dtype=ids.dtype), ids])
                attn = torch.cat([torch.zeros(pad, dtype=attn.dtype), attn])
            out_ids.append(ids)
            out_attn.append(attn)
        return torch.stack(out_ids), torch.stack(out_attn)
