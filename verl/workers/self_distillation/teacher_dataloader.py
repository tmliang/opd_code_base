# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Base classes for OPSD teacher dataloaders.

A *teacher dataloader* takes one student-facing sample and returns the
*teacher-facing* chat messages (and optional multi-modal payload). Subclass
**exactly one** of:

* :class:`OfflineTeacherDataloader` — output depends only on the original
  dataset row (e.g. inject a gold answer / reference solution, swap an image,
  rewrite the system prompt). Cheap; can be applied before rollout.
* :class:`OnlineTeacherDataloader` — output also depends on the student
  rollout (response text, reward, sibling rollouts). Required for OPSD-style
  recipes that paste the best sibling response as a demonstration.

Both subclasses must override :meth:`build_one` and may write any chat
template / formatting logic directly inside the class — the framework will
re-apply ``tokenizer.apply_chat_template`` (or the multi-modal processor)
exactly as the student-side dataloader does.

The user only points the config at the class FQN (offline vs online is
auto-detected via ``isinstance``)::

    self_distill:
      dataloader: my_pkg.my_module:MyTeacherDataloader
      dataloader_kwargs:
        some_param: ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


@dataclass
class TeacherSample:
    """One teacher-side example.

    Attributes:
        messages: OpenAI-style chat messages — **same schema verl's
            ``RLHFDataset.raw_prompt`` produces and that the student consumed
            at rollout time**. For multi-modal tasks, content segments inline
            the media (e.g. ``{"type": "image", "image": <PIL.Image>}``); the
            framework will re-apply ``processor.apply_chat_template`` / the
            HuggingFace processor on this list, exactly like the actor side.
            Do not pre-tokenize.
        multi_modal_data: Optional dict using verl's plural keys
            ``{"images": [...], "videos": [...], "audios": [...]}`` — same
            shape ``AgentLoop`` / ``RLHFDataset`` produce. Pass through
            unchanged if you only rewrote the text part; supply your own
            payload if you swapped images / added crops / etc.
        skip: When ``True`` the framework zeros out the distillation term
            for this sample (policy loss still applies). Use when a
            meaningful teacher prompt cannot be constructed.
    """

    messages: list[dict] = field(default_factory=list)
    multi_modal_data: Optional[dict] = None
    skip: bool = False


@dataclass
class BatchView:
    """Read-only view over the current batch.

    Passed to :class:`OnlineTeacherDataloader` so subclasses can aggregate
    across samples — e.g. find sibling rollouts that share the same UID for
    OPSD-style best-of-N self-demonstration.
    """

    uids: list[Any] = field(default_factory=list)
    rewards: Optional[list[float]] = None
    response_texts: Optional[list[str]] = None
    extra: dict[str, list[Any]] = field(default_factory=dict)

    def iter_same_uid(self, index: int) -> Iterable[int]:
        """Yield indices of all other samples sharing this sample's UID."""
        if not self.uids:
            return
        target = self.uids[index]
        for j, u in enumerate(self.uids):
            if j != index and u == target:
                yield j


class TeacherDataloader(ABC):
    """Common base. Do not subclass directly — pick offline xor online."""

    def __init__(self, *, tokenizer, processor=None, **kwargs):
        self.tokenizer = tokenizer
        self.processor = processor
        #: free-form config block forwarded from ``self_distill.dataloader_kwargs``.
        self.config = kwargs


class OfflineTeacherDataloader(TeacherDataloader):
    """Build the teacher sample from the *original* row only.

    Override :meth:`build_one`. Define any chat template inside the class
    (e.g. as a class constant or in ``__init__``).
    """

    @abstractmethod
    def build_one(
        self,
        *,
        prompt_messages: Optional[list[dict]],
        prompt_text: str,
        multi_modal_data: Optional[dict],
        extra_info: dict,
    ) -> TeacherSample:
        """Return the teacher sample for one row.

        Args:
            prompt_messages: The dataset's ``raw_prompt`` for this sample —
                the **chat messages list with embedded multi-modal content
                segments** that the student consumed at rollout time. This is
                the canonical input; build your teacher messages by editing
                a copy of it. ``None`` only when the rollout dropped it,
                which is rare — prefer raising or returning
                ``TeacherSample(skip=True)`` over fabricating fake messages.
            prompt_text: Decoded student prompt token IDs (special tokens
                included). Provided as a debugging aid; do **not** wrap it
                in a synthetic ``{"role": "user", "content": prompt_text}``
                message — that does not round-trip through the chat
                template cleanly.
            multi_modal_data: ``{"images": [...], "videos": [...],
                "audios": [...]}`` (verl's plural-key convention) or
                ``None``. Pass through unchanged if you only rewrote text.
            extra_info: The dataset row's ``extra_info`` dict (gold answer,
                ``data_source``, annotations, ...).
        """


class OnlineTeacherDataloader(TeacherDataloader):
    """Build the teacher sample from the original row **plus** the rollout output.

    Override :meth:`build_one`. Use ``batch_view`` to reach across samples
    if your recipe needs sibling rollouts.
    """

    @abstractmethod
    def build_one(
        self,
        *,
        prompt_messages: Optional[list[dict]],
        prompt_text: str,
        response_text: str,
        reward: Optional[float],
        multi_modal_data: Optional[dict],
        extra_info: dict,
        batch_view: BatchView,
        index: int,
    ) -> TeacherSample:
        """Return the teacher sample for one row.

        ``prompt_messages`` / ``prompt_text`` / ``multi_modal_data`` /
        ``extra_info`` follow the same conventions as
        :meth:`OfflineTeacherDataloader.build_one`.

        Args:
            response_text: Decoded student response token IDs.
            reward: Scalar reward for this rollout (``None`` if unavailable).
            batch_view: Read-only view of the whole batch — use
                ``batch_view.iter_same_uid(index)`` to find sibling
                rollouts under the same prompt UID.
            index: This sample's index inside ``batch_view``.
        """
