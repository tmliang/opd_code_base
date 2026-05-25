# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SDPO teacher dataloaders (sibling-rollout reprompt + identity baseline)."""

from __future__ import annotations

import copy
import re

from verl.workers.self_distillation import (
    OfflineTeacherDataloader,
    OnlineTeacherDataloader,
    TeacherSample,
)

from ._utils import last_user_message
from . import register


@register("sdpo")
class sdpo_dataloader(OnlineTeacherDataloader):  # noqa: N801 (lowercase per user request)
    """Faithful SDPO reprompting recipe.

    For each student rollout, find a *successful* sibling rollout under the
    same UID (reward ``>= success_reward_threshold``). If one exists, build
    the teacher prompt by formatting the SDPO reprompt template::

        {prompt}{solution_section}\\n\\nCorrectly solve the original question.

    with the sibling's response substituted into ``solution_section`` via
    ``solution_template``. Samples with no successful sibling are skipped so
    the SDPO loss is masked out — same semantics as upstream
    ``RayPPOTrainer._maybe_build_self_distillation_batch``'s
    ``self_distillation_mask``.

    YAML knobs (``self_distill.dataloader_kwargs``):

    - ``success_reward_threshold`` (float, default 1.0): min sequence reward
      to qualify as a successful demonstration.
    - ``dont_reprompt_on_self_success`` (bool, default False): if True, do
      not splice a demonstration into samples that themselves succeeded.
    - ``remove_thinking_from_demonstration`` (bool, default False): strip
      ``<think>...</think>`` blocks from the sibling response.
    - ``reprompt_template`` (str): uses ``{prompt}`` / ``{solution}`` placeholders.
    - ``solution_template`` (str): wraps the sibling response.
    """

    DEFAULT_REPROMPT_TEMPLATE = (
        "{prompt}{solution}\n\n"
        "Correctly solve the original question.\n"
    )
    DEFAULT_SOLUTION_TEMPLATE = (
        "\n"
        "Correct solution:\n\n"
        "{successful_previous_attempt}\n\n"
    )

    def __init__(self, *, tokenizer, processor=None, **kwargs):
        super().__init__(tokenizer=tokenizer, processor=processor, **kwargs)
        self._success_threshold = float(self.config.get("success_reward_threshold", 1.0))
        self._dont_reprompt_on_self_success = bool(
            self.config.get("dont_reprompt_on_self_success", False)
        )
        self._strip_thinking = bool(
            self.config.get("remove_thinking_from_demonstration", False)
        )
        self._reprompt_template = str(
            self.config.get("reprompt_template", self.DEFAULT_REPROMPT_TEMPLATE)
        )
        self._solution_template = str(
            self.config.get("solution_template", self.DEFAULT_SOLUTION_TEMPLATE)
        )

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

    def _pick_solution(self, *, index, reward, batch_view):
        rewards = batch_view.rewards or []
        responses = batch_view.response_texts or []
        if not rewards or not responses:
            return None

        self_success = reward is not None and reward >= self._success_threshold
        for j in batch_view.iter_same_uid(index):
            if self._dont_reprompt_on_self_success and j == index:
                continue
            rj = rewards[j] if j < len(rewards) else None
            if rj is None or rj < self._success_threshold:
                continue
            demo = responses[j] if j < len(responses) else None
            if not demo:
                continue
            return self._strip_think_tags(demo) if self._strip_thinking else demo

        # Fall back to self if it succeeded (matches upstream ``solution_idxs``
        # building, which includes the current sample unless
        # ``dont_reprompt_on_self_success`` is set).
        if self_success and not self._dont_reprompt_on_self_success:
            demo = responses[index] if index < len(responses) else None
            if demo:
                return self._strip_think_tags(demo) if self._strip_thinking else demo
        return None

    def build_one(
        self,
        *,
        prompt_messages,
        prompt_text,
        response_text,
        reward,
        multi_modal_data,
        extra_info,
        batch_view,
        index,
    ):
        if prompt_messages is None:
            return TeacherSample(messages=[], skip=True)

        solution = self._pick_solution(index=index, reward=reward, batch_view=batch_view)
        if solution is None:
            return TeacherSample(messages=[], skip=True)

        messages = copy.deepcopy(prompt_messages)
        user = last_user_message(messages)
        if user is None or not isinstance(user.get("content"), str):
            return TeacherSample(messages=[], skip=True)

        solution_section = self._solution_template.format(
            successful_previous_attempt=solution
        )
        user["content"] = self._reprompt_template.format(
            prompt=user["content"],
            solution=solution_section,
            feedback="",
        )
        return TeacherSample(messages=messages, multi_modal_data=multi_modal_data)


@register("sdpo_identity")
class sdpo_identity_dataloader(OfflineTeacherDataloader):  # noqa: N801
    """Identity rewrite: teacher prompt == student prompt.

    NOT the original SDPO algorithm — this is the pure EMA/ref-KL baseline
    where the only teacher signal is the alpha-KL against a (possibly EMA)
    reference policy. Useful as an ablation / sanity baseline.
    """

    def build_one(self, *, prompt_messages, prompt_text, multi_modal_data, extra_info):
        if prompt_messages is None:
            return TeacherSample(messages=[], skip=True)
        return TeacherSample(
            messages=copy.deepcopy(prompt_messages),
            multi_modal_data=multi_modal_data,
        )
