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
"""Multiple-choice (A/B/C/D) accuracy for vision benchmarks (V*Bench, Vision-OPD, ...).

Follows the V* evaluation convention (penghao-wu/vstar): the model is
prompted to answer with the option's letter; we extract the final letter
from the response (scanning from the tail so CoT noise is ignored) and
compare against ``ground_truth``.
"""

from __future__ import annotations

import re

_BOXED_RE = re.compile(r"\\boxed\{\s*\(?\s*([A-Z])\s*\)?\s*\}")
_ANSWER_RE = re.compile(r"answer\s*(?:is|:)\s*\*{0,2}\s*\(?\s*([A-Z])\b", re.IGNORECASE)
_OPTION_RE = re.compile(r"\boption\s*\(?\s*([A-Z])\b", re.IGNORECASE)
_PAREN_RE = re.compile(r"\(\s*([A-Z])\s*\)")
_ISOLATED_RE = re.compile(r"(?:^|[\s>])([A-Z])(?=[\s.,;:!?)\]\"']|$)")


def _extract_letter(text: str):
    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t:
        return None
    tail = t.rsplit("</think>", 1)[-1].strip() or t
    for regex in (_BOXED_RE, _ANSWER_RE, _OPTION_RE, _PAREN_RE, _ISOLATED_RE):
        matches = regex.findall(tail)
        if matches:
            return matches[-1].upper()
    if len(tail) == 1 and tail.isalpha():
        return tail.upper()
    for regex in (_BOXED_RE, _ANSWER_RE, _OPTION_RE, _PAREN_RE, _ISOLATED_RE):
        matches = regex.findall(t)
        if matches:
            return matches[-1].upper()
    return None


def compute_score(solution_str, ground_truth, extra_info=None):
    gold = (ground_truth or "").strip().upper()
    pred = _extract_letter(solution_str) or ""
    acc = 1.0 if pred and pred == gold else 0.0
    return {"score": acc, "acc": acc, "pred": pred, "gold": gold}
