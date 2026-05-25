# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""OPSD teacher dataloaders (paper recipe + lightweight hint variant)."""

from __future__ import annotations

import copy
from typing import Any

from verl.workers.self_distillation import OfflineTeacherDataloader, TeacherSample

from ._utils import last_user_message
from . import register


@register("opsd")
class opsd_dataloader(OfflineTeacherDataloader):  # noqa: N801 (lowercase per user request)
    """OPSD recipe — faithful reproduction of ``siyan-zhao/OPSD`` ``data_collator.py``
    (non-``reason_first`` path).

    The original OPSD trainer rewrites the user turn as::

        Problem: {problem}

        Here is a reference solution to this problem:
        === Reference Solution Begin ===
        {solution}
        === Reference Solution End ===
        {transition_prompt}
        Please reason step by step, and put your final answer within \\boxed{{}}.

    See ``OPSD/data_collator.py::SelfDistillationDataCollator``.

    Reference solution is picked from the first non-empty of
    ``extra_info["solution"]``, ``extra_info["reference_solution"]``,
    ``extra_info["answer"]`` (GSM8K style). The original user message text is
    treated as the ``Problem:`` body (any leading ``Problem:`` prefix is
    de-duplicated).

    YAML knobs (``self_distill.dataloader_kwargs``):

    - ``solution_field`` (str, optional): override the ``extra_info`` field to
      read the reference solution from.
    - ``transition_prompt`` (str): override the transition prompt verbatim.
    - ``final_instruction`` (str): override the trailing
      ``Please reason step by step…`` line.
    """

    DEFAULT_TRANSITION_PROMPT = (
        "\n\nAfter reading the reference solution above, make sure you truly understand "
        "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
        "own words and independent reasoning, derive the same final answer to the problem above. "
        "Think step by step, explore different approaches, and don't be afraid to backtrack "
        "or reconsider if something doesn't work out:\n"
    )
    DEFAULT_FINAL_INSTRUCTION = (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    DEFAULT_SOLUTION_FIELDS = ("solution", "reference_solution", "answer")

    def __init__(self, *, tokenizer, processor=None, **kwargs):
        super().__init__(tokenizer=tokenizer, processor=processor, **kwargs)
        self._transition_prompt = str(
            self.config.get("transition_prompt", self.DEFAULT_TRANSITION_PROMPT)
        )
        self._final_instruction = str(
            self.config.get("final_instruction", self.DEFAULT_FINAL_INSTRUCTION)
        )
        sf = self.config.get("solution_field")
        self._solution_fields = (sf,) if sf else self.DEFAULT_SOLUTION_FIELDS

    def _pick_solution(self, extra_info: dict[str, Any] | None) -> str | None:
        if not extra_info:
            return None
        for k in self._solution_fields:
            v = extra_info.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return None

    @staticmethod
    def _strip_problem_prefix(text: str) -> str:
        stripped = text.lstrip()
        for prefix in ("Problem:", "problem:", "Question:", "question:"):
            if stripped.startswith(prefix):
                return stripped[len(prefix):].lstrip()
        return text

    def build_one(self, *, prompt_messages, prompt_text, multi_modal_data, extra_info):
        if prompt_messages is None:
            return TeacherSample(messages=[], skip=True)
        solution = self._pick_solution(extra_info or {})
        if solution is None:
            return TeacherSample(messages=[], skip=True)
        messages = copy.deepcopy(prompt_messages)
        user = last_user_message(messages)
        if user is None or not isinstance(user.get("content"), str):
            return TeacherSample(messages=[], skip=True)
        problem_body = self._strip_problem_prefix(user["content"])
        user["content"] = (
            f"Problem: {problem_body}\n\n"
            f"Here is a reference solution to this problem:\n"
            f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
            f"{self._transition_prompt}\n"
            f"{self._final_instruction}"
        )
        return TeacherSample(messages=messages, multi_modal_data=multi_modal_data)


@register("opsd_hint")
class opsd_hint_dataloader(OfflineTeacherDataloader):  # noqa: N801
    """Lightweight OPSD variant: append ``(Hint: the final answer is X.)`` to
    the last user turn. Useful when only the final answer (not a full reference
    solution) is available. NOT the original OPSD paper recipe.
    """

    HINT_SUFFIX = "\n\n(Hint: the final answer is {answer}. Show your reasoning step by step.)"

    def build_one(self, *, prompt_messages, prompt_text, multi_modal_data, extra_info):
        answer = extra_info.get("answer") if extra_info else None
        if answer is None or prompt_messages is None:
            return TeacherSample(messages=[], skip=True)
        messages = copy.deepcopy(prompt_messages)
        user = last_user_message(messages)
        if user is None:
            return TeacherSample(messages=[], skip=True)
        user["content"] = (user.get("content") or "") + self.HINT_SUFFIX.format(answer=answer)
        return TeacherSample(messages=messages, multi_modal_data=multi_modal_data)
